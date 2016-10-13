from __future__ import print_function, division
import tensorflow as tf
from collections import namedtuple
import tensorflow.contrib.slim as slim

from classification.model import ModelBase, TrainNetInfo

from .tf_layers import conv1d_layer, dense_layer, misconception_layer, dropout_layer
from .tf_layers import batch_norm

TowerParams = namedtuple("TowerParams",
                         ["filter_count", "filter_widths", "pool_size",
                          "pool_stride", "keep_prob"])


class Model(ModelBase):

    initial_learning_rate = 0.1
    learning_decay_rate = 0.99
    decay_examples = 10000
    momentum = 0.9

    tower_params = [
        TowerParams(*x)
        for x in [(16, [3], 3, 2, 1.0)] * 9 + [(16, [3], 3, 2, 0.8)]
    ]

    @property
    def window_max_points(self):
        length = 1
        for tp in reversed(self.tower_params):
            length = length * tp.pool_stride + (tp.pool_size - tp.pool_stride)
            length += sum(tp.filter_widths) - len(tp.filter_widths)
        return length

    def build_model(self, is_training, current):

        # Build a tower consisting of stacks of misconception layers in parallel
        # with size 1 convolutional shortcuts to help train.

        for i, tp in enumerate(self.tower_params):
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
                    shunt = tf.nn.elu(
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
            logits = dense_layer(current, self.num_classes)

        return logits

    def build_inference_net(self, features):
        return self.build_model(tf.constant(False), features)

    def build_training_net(self, features, labels, fishing_timeseries_labels):
        logits = self.build_model(tf.constant(True), features)
        example = slim.get_or_create_global_step() * self.batch_size
        #
        with tf.variable_scope('training'):
            # Decay the learning rate by `learning_decay_rate` every
            # `decay_examples`.
            learning_rate = tf.train.exponential_decay(
                self.initial_learning_rate, example, self.decay_examples,
                self.learning_decay_rate)

            # Compute loss and predicted probabilities
            with tf.name_scope('loss-function'):
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                   labels))
                # Use simple momentum for the optimization.
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                       self.momentum)

        return TrainNetInfo(loss, optimizer, logits)
