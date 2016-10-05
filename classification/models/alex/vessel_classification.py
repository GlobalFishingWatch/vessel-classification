from __future__ import absolute_import
import argparse
import json
from . import layers
import logging
import math
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model:

    feature_duration_days = 180
    num_classes = 9
    num_feature_dimensions = 9
    max_sample_frequency_seconds = 5 * 60
    max_window_duration_seconds = feature_duration_days * 24 * 3600

    # We allocate a much smaller buffer than would fit the specified time
    # sampled at 5 mins intervals, on the basis that the sample is almost
    # always much more sparse.
    window_max_points = (max_window_duration_seconds /
                         max_sample_frequency_seconds) / 4
    window_size = 3
    stride = 2
    feature_depth = 20
    levels = 10
    batch_size = 32
    min_viable_timeslice_length = 500

    def zero_pad_features(self, features):
        """ Zero-pad features in the depth dimension to match requested feature depth. """

        feature_pad_size = self.feature_depth - self.num_feature_dimensions
        assert (feature_pad_size >= 0)
        zero_padding = tf.zeros([self.batch_size, 1, self.window_max_points,
                                 feature_pad_size])
        padded = tf.concat(3, [features, zero_padding])

        return padded

    def build_training_net(self, features, labels):

        features = self.zero_pad_features(features)
        one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)

        logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.num_classes, True)

        loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)

        optimizer = tf.train.AdamOptimizer(2e-5)

        return loss, optimizer, logits

    def build_inference_net(self, features):

        features = self.zero_pad_features(features)

        logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.num_classes, tf.constant(False))

        return logits
