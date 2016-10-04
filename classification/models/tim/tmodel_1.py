from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import time
import os
import sys
from itertools import islice, count
from tensorflow.core.framework import summary_pb2
from collections import namedtuple
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import time
from classification import utility
import logging
from .tf_layers import conv1d_layer, dense_layer, misconception_layer, dropout_layer
from .tf_layers import batch_norm, leaky_rectify

TowerParams = namedtuple("TowerParams",
                         ["filter_count", "filter_widths", "pool_size",
                          "pool_stride", "keep_prob"])


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(
        tag=name, simple_value=val)])


GraphData = namedtuple("GraphData", ["X", "y", "Y_", "logits", "is_training",
                                     "batch_size", "optimizer"])



class Trainer:
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_MAX_EXAMPLES = 30000 
    N_FEATURES = 9
    N_LOADED_FEATURES = 9
    INITIAL_LEARNING_RATE = 0.1
    INITIAL_MOMENTUM = 0.9
    DECAY_RATE = 0.98
    N_SAMPLES = 1
    N_FOLDS = 5
    WHICH_FOLD = 0
    SAVE_INTERVAL_S = 30.0
    VALIDATION_POINTS = 1024
    NUM_PARALLEL_READERS = 24
    MAX_WINDOW_DURATION_SECONDS = 60 * 60 * 24 * 180

    _trainable_parameters = None

    N_CATS = 10

    TOWER_PARAMS = [
        TowerParams(*x)
        for x in [(16, [3], 3, 2, 1.0)] * 9 + [(16, [3], 3, 2, 0.8)]
    ]

    INIT_KEEP_PROB = 1.0

    def __init__(self, root_feature_path, training_output_path):
        self.CHECKPOINT_DIR = training_output_path
        self.BASE_FEATURE_PATH = root_feature_path

    @property
    def SEQ_LENGTH(self):
        length = 1
        for tp in reversed(self.TOWER_PARAMS):
            length = length * tp.pool_stride + (tp.pool_size - tp.pool_stride)
            length += sum(tp.filter_widths) - len(tp.filter_widths)
        return length

    # TODO: remove X, y (they are there for backwards compatibility during transisiton)
    def build_model(self, is_training, X):

        y = tf.placeholder(tf.int64, shape=(None, ), name="labels") # XXX GET RID OF?

        current = X

        if self.INIT_KEEP_PROB < 1:
            dropout_layer(current, is_training, self.INIT_KEEP_PROB)

        # Build a tower consisting of stacks of misconception layers in parallel
        # with size-1 convolutional shortcuts to help train.
        for i, tp in enumerate(self.TOWER_PARAMS):
            with tf.variable_scope('tower-segment-{}'.format(i + 1)):

                # Misconception stack
                mc = current
                for j, w in enumerate(tp.filter_widths):
                    mc = misconception_layer(
                        mc,
                        tp.filter_count,
                        is_training,
                        filter_size=w,
                        padding="VALID",
                        name='misconception-{}'.format(j))

                # Build a shunt layer (resnet) to help convergence
                with tf.variable_scope('shunt'):
                    # Trim current before making the skip layer so that it matches the dimensons of
                    # the mc stack
                    W = int(current.get_shape().dims[2])
                    delta = sum(tp.filter_widths) - len(tp.filter_widths)
                    shunt = tf.slice(current, [0, 0, delta // 2, 0],
                                     [-1, -1, W - delta, -1])
                    shunt = leaky_rectify(
                        batch_norm(
                            conv1d_layer(shunt, 1, tp.filter_count),
                            is_training))

                current = shunt + mc

                current = tf.nn.max_pool(
                    current, [1, 1, tp.pool_size, 1],
                    [1, 1, tp.pool_stride, 1],
                    padding="VALID")
                if tp.keep_prob < 1:
                    current = dropout_layer(current, is_training, tp.keep_prob)

        # Remove extra dimensions
        H, W, C = [int(x) for x in current.get_shape().dims[1:]]
        current = tf.reshape(current, (-1, C))

        # Determine fishing estimate
        with tf.variable_scope("prediction-layer"):
            logits = dense_layer(current, self.N_CATS)
            Y_ = tf.nn.softmax(logits)

        #
        return X, y, Y_, logits

    @property
    def CHECKPOINT_PATH(self):
        return os.path.join(self.CHECKPOINT_DIR,
                            self.__class__.__name__ + '.ckpt')

    def _set_trainable_parameters(self):
        trainable = 0
        for var in tf.trainable_variables():
            trainable += np.prod([dim.value for dim in var.get_shape()])
        self._trainable_parameters = trainable

    @property
    def trainable_parameters(self):
        if self._trainable_parameters is None:
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    self.build_test_graph(sess, None)
                    self._set_trainable_parameters()
        return self._trainable_parameters

    def eval_in_batches(self, size, sess, gdata):
        """Get all predictions for a dataset by running it in small batches."""
        predictions = np.ndarray(shape=(size, self.N_CATS), dtype=np.float32)
        labels = np.ndarray(shape=(size, ), dtype=np.int32)
        feed_dict = {gdata.is_training: False}
        for begin in xrange(0, size, gdata.batch_size):
            end = min(begin + gdata.batch_size, size)
            count = end - begin
            logging.debug("EVAL IN BATCHES %s, %s, %s", begin, end, count)
            sys.stdout.flush()
            raw_preds, raw_labels = sess.run([gdata.Y_, gdata.y],
                                                   feed_dict=feed_dict)
            predictions[begin:end, :] = raw_preds[:count]
            labels[begin:end] = raw_labels[:count]
        return predictions, labels

    def predict_proba_from_src(self, src, trials=1, batch_size=None):
        _, X, _ = src.iters(self.N_SAMPLES)
        size = len(src)
        return self.predict_proba(X, size, trials, batch_size)

    def build_test_graph(self, batch_size, X, y):

        is_training = tf.placeholder(tf.bool)

        _, _, Y_, logits = self.build_model(is_training, X)
        self._set_trainable_parameters()

        with tf.variable_scope('training'):
            batch = tf.Variable(0, trainable=False, name="batch")

        return GraphData(
            X, y, Y_, logits, is_training, batch_size,
            optimizer=None), batch

    def build_train_graph(self, batch_size, X, y, train_size=20000):

        is_training = tf.placeholder(tf.bool)
        _, _, Y_, logits = self.build_model(is_training, X)
        #
        self._set_trainable_parameters()
        #
        with tf.variable_scope('training'):

            # Optimizer: set up a variable that's incremented once per batch and
            # controls the learning rate decay.
            batch = tf.Variable(0, trainable=False, name="batch")
            #
            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate = tf.train.exponential_decay(
              self.INITIAL_LEARNING_RATE,
              batch * batch_size,  # Current index into the dataset.
              train_size,          # Decay step.
              self.DECAY_RATE,
              staircase=True)

            # Compute loss and predicted probabilities `Y_`
            with tf.name_scope('loss-function'):
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
                # Use simple momentum for the optimization.
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                       0.9).minimize(
                                                           loss,
                                                           global_step=batch)

            predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
            accuracy = slim.metrics.accuracy(y, predictions)

            tf.scalar_summary('Training loss', loss)
            tf.scalar_summary('Training accuracy', accuracy)
            tf.scalar_summary('Learning rate', learning_rate)

        return GraphData(X, y, Y_, logits, is_training, batch_size,
                         optimizer), batch

    def run_training(self, target="", is_chief=True):

        input_file_pattern = self.BASE_FEATURE_PATH + '/Training/shard-*-of-*.tfrecord'
        features, labels = self.data_reader(input_file_pattern, True)

        batch_size = self.DEFAULT_BATCH_SIZE

        gdata, batch = self.build_train_graph(batch_size, features, labels)

        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(self.CHECKPOINT_DIR)
        saver = tf.train.Saver()

        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=self.CHECKPOINT_DIR,
            init_op=tf.initialize_all_variables(),
            saver=saver,
            summary_op=None,
            global_step=batch,
            save_model_secs=60)

        with sv.managed_session(target) as sess:


            for step in count():

                # XXX could now built this into train / test dics!
                feed_dict = {
                    gdata.is_training: True
                }  

                summary, predicted, _ = sess.run(fetches=[merged, gdata.Y_,
                                                          gdata.optimizer],
                                                 feed_dict=feed_dict)

                if is_chief:
                    writer.add_summary(summary, step)

                if sv.should_stop(
                ): 
                    break


            writer.flush()
            sv.stop()

        return self

    def run_evaluation(self, target="", timeout=60 * 15):

        input_file_pattern = self.BASE_FEATURE_PATH + '/Test/shard-*-of-*.tfrecord'
        features, labels = self.data_reader(input_file_pattern, False)

        batch_size = self.DEFAULT_BATCH_SIZE

        gdata, batch = self.build_test_graph(batch_size, features,
                                             labels)

        logdir = self.CHECKPOINT_DIR + "_logs"

        writer = tf.train.SummaryWriter(logdir)
        saver = tf.train.Saver()

        sv = tf.train.Supervisor(
            graph=ops.get_default_graph(),
            logdir=logdir,
            summary_op=None,
            summary_writer=None,
            init_op=tf.initialize_all_variables(),
            global_step=None,
            saver=saver,
            save_summaries_secs=0)

        for checkpoint_path in slim.evaluation.checkpoints_iterator(
                    self.CHECKPOINT_DIR, timeout=timeout):

            logging.debug('Starting eval')
            sys.stdout.flush()

            with sv.managed_session(
                    target, start_standard_services=False,
                    config=None) as sess:

                try:
                    sv.saver.restore(sess, checkpoint_path)
                except:
                    logging.warning("Could not restore from %s %s",
                                    checkpoint_path, "skipping")
                    continue
                sv.start_queue_runners(sess)

                batch_val = sess.run(batch)

                logging.debug("Eval step: %s", batch_val)
                preds, labels = self.eval_in_batches(self.VALIDATION_POINTS,
                                                     sess,
                                                     gdata)
                accuracy = (labels == np.argmax(preds, axis=1)).mean()
                writer.add_summary(
                    make_summary("Test accuracy", accuracy), batch_val)
                #
                writer.flush()

                if batch_val * batch_size > self.DEFAULT_MAX_EXAMPLES or sv.should_stop():
                    logging.info("Ending: %s", sv.should_stop())
                    sys.stdout.flush()
                    break



    def data_reader(self, input_file_pattern, is_training):
        matching_files_i = tf.matching_files(input_file_pattern)
        matching_files = tf.Print(matching_files_i, [matching_files_i],
                                  "Files: ")
        filename_queue = tf.train.input_producer(matching_files, shuffle=True)
        capacity = 600
        min_size_after_deque = capacity - self.DEFAULT_BATCH_SIZE * 4

        max_replication = 8

        readers = []
        for _ in range(self.NUM_PARALLEL_READERS):
            readers.append(
                utility.cropping_weight_replicating_feature_file_reader(
                    filename_queue, self.N_LOADED_FEATURES + 1,
                    self.MAX_WINDOW_DURATION_SECONDS, self.SEQ_LENGTH, 500,
                    max_replication))

        features, time_bounds, labels = tf.train.shuffle_batch_join(
            readers,
            self.DEFAULT_BATCH_SIZE,
            capacity,
            min_size_after_deque,
            enqueue_many=True,
            shapes=[[1, self.SEQ_LENGTH, self.N_FEATURES], [2], []])

        return features, labels
