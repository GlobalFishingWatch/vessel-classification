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
from classification.model import ModelBase, TrainNetInfo
import logging
from .tf_layers import conv1d_layer, dense_layer, misconception_layer, dropout_layer
from .tf_layers import batch_norm, leaky_rectify

TowerParams = namedtuple("TowerParams",
                         ["filter_count", "filter_widths", "pool_size",
                          "pool_stride", "keep_prob"])


class Model(ModelBase):
    DEFAULT_BATCH_SIZE = 32
    N_FEATURES = 9
    INITIAL_LEARNING_RATE = 0.1
    INITIAL_MOMENTUM = 0.9
    DECAY_RATE = 0.98
    MAX_WINDOW_DURATION_SECONDS = 60 * 60 * 24 * 180

    _trainable_parameters = None

    N_CATS = 10

    TOWER_PARAMS = [
        TowerParams(*x)
        for x in [(16, [3], 3, 2, 1.0)] * 9 + [(16, [3], 3, 2, 0.8)]
    ]

    INIT_KEEP_PROB = 1.0

    batch_size = DEFAULT_BATCH_SIZE
    num_feature_dimensions = N_FEATURES
    max_window_duration_seconds = MAX_WINDOW_DURATION_SECONDS
    min_viable_timeslice_length = 500

    @property
    def SEQ_LENGTH(self):
        length = 1
        for tp in reversed(self.TOWER_PARAMS):
            length = length * tp.pool_stride + (tp.pool_size - tp.pool_stride)
            length += sum(tp.filter_widths) - len(tp.filter_widths)
        return length

    window_max_points = SEQ_LENGTH

    def build_model(self, is_training, X):

        current = X

        # Build a tower consisting of stacks of misconception layers in parallel
        # with size 1 convolutional shortcuts to help train.

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
            logits = dense_layer(current, self.N_CATS)

        return logits

    def build_inference_net(self, features):

        return self.build_model(tf.constant(False), features)

    def build_training_net(self, features, labels):

        logits = self.build_model(tf.constant(True), features)
        # TODO [bitsofbits]
        # leftover from when actually knew training size so could drop size once per
        # epoch. Replace with some sort of sensible decay;
        train_size = 20000
        #
        batch = slim.get_or_create_global_step()
        #
        with tf.variable_scope('training'):
            # Optimizer: set up a variable that's incremented once per batch and
            # controls the learning rate decay.
            #
            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate = tf.train.exponential_decay(
              self.INITIAL_LEARNING_RATE,
              batch * self.DEFAULT_BATCH_SIZE,  # Current index into the dataset.
              train_size,          # Decay step.
              self.DECAY_RATE,
              staircase=True)

            # Compute loss and predicted probabilities `Y_`
            with tf.name_scope('loss-function'):
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                   labels))
                # Use simple momentum for the optimization.
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

        return TrainNetInfo(loss, optimizer, logits)
