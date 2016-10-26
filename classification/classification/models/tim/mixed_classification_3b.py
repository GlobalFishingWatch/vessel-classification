from __future__ import print_function, division
import tensorflow as tf
from collections import namedtuple
import tensorflow.contrib.slim as slim

from classification.model import MixedFishingModelBase, TrainNetInfo

from .tf_layers import conv1d_layer, dense_layer, misconception_layer, dropout_layer
from .tf_layers import batch_norm


TowerParams = namedtuple("TowerParams",
                         ["filter_count", "filter_widths", "pool_size", "pool_stride",
                          "keep_prob", "shunt"])



class Model(MixedFishingModelBase):

    initial_learning_rate = 0.01
    learning_decay_rate = 0.9
    decay_examples = 10000
    momentum = 0.9

    tower_params = [
        TowerParams(*x)
        for x in 
        [(24, [(3, 1), (3, 2), (3, 4), (3, 8)], 3, 2, 1.0, False)] * 1 + 
                 [(16, [(3, 1)],                 3, 2, 1.0, True)] * 8 + 
                 [(16, [(3, 1)],                 3, 2, 0.8, True)]
    ]

    @property
    def window_max_points(self):
        length = 1
        for tp in reversed(self.tower_params):
            length = length * tp.pool_stride + (tp.pool_size - tp.pool_stride)
        return length

    def build_model(self, is_training, current):

        # Build a tower consisting of stacks of misconception layers in parallel
        # with size 1 convolutional shortcuts to help train.

        for i, tp in enumerate(self.tower_params):
            with tf.variable_scope('tower-segment-{}'.format(i + 1)):

                # Misconception stack
                mc = current
                for j, (w, r) in enumerate(tp.filter_widths):
                    mc = misconception_layer(
                        mc,
                        tp.filter_count,
                        is_training,
                        filter_size=w,
                        filter_rate=r, 
                        padding="SAME",
                        name='misconception-{}'.format(j))

                if tp.shunt:
                    # Build a shunt layer (resnet) to help convergence
                    with tf.variable_scope('shunt'):
                        # Trim current before making the skip layer so that it matches the dimensons of
                        # the mc stack
                        shunt = tf.nn.elu(
                            batch_norm(
                                conv1d_layer(current, 1, tp.filter_count),
                                is_training))
                    current = shunt + mc
                else:
                    current = mc

                if i == 0:
                    # Stash first layer away as input fishing classifier
                    fishing_classifier_input = current

                current = tf.nn.max_pool(
                    current, [1, 1, tp.pool_size, 1],
                    [1, 1, tp.pool_stride, 1],
                    padding="VALID")
                if tp.keep_prob < 1:
                    current = dropout_layer(current, is_training, tp.keep_prob)

        # Remove extra dimensions
        H, W, C = [int(x) for x in current.get_shape().dims[1:]]
        current = tf.reshape(current, (-1, C))

        # Determine classification logits
        with tf.variable_scope("prediction-layer"):
            class_logits = dense_layer(current, self.num_classes)

        # Assemble the fishing score logits

        embedding = tf.reshape(current, (-1, 1, 1, C))
        embedding = tf.tile(embedding, [1, 1, self.window_max_points, 1])
        current = tf.concat(3, [fishing_classifier_input, embedding])
        current = tf.nn.elu(
                        batch_norm(
                            conv1d_layer(current, 1, 1024, name="fishing1"),
                            is_training))
        current = conv1d_layer(current, 1, 1, name="fishing_logits")
        fishing_logits = tf.reshape(current, (-1, self.window_max_points))

        return class_logits, fishing_logits

    def build_inference_net(self, features):
        return self.build_model(tf.constant(False), features)

    def build_training_net(self, features, labels, fishing_timeseries_labels):
        vessel_class_logits, fishing_logits = self.build_model(tf.constant(True), features)
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
                class_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        vessel_class_logits, labels))
                fishing_mask = tf.to_float(tf.not_equal(fishing_timeseries_labels, -1))
                fishing_targets = tf.to_float(fishing_timeseries_labels > 0.5)
                fishing_loss = (tf.reduce_sum(fishing_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                    fishing_logits, fishing_targets)) /
                         (100 + tf.reduce_sum(fishing_mask))) # TODO: no magic numbers

                loss  = 0.01 * class_loss + fishing_loss
                # Use simple momentum for the optimization.
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                       self.momentum)

        return TrainNetInfo(loss, optimizer, vessel_class_logits, fishing_logits)
