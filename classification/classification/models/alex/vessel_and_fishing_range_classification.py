from __future__ import absolute_import
import argparse
import json
from . import layers
from classification.model import ModelBase, TrainNetInfo
import logging
import math
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model(ModelBase):

    window_size = 3
    stride = 2
    feature_depth = 50
    levels = 10

    def misconception_with_fishing_ranges(self, input, num_classes,
                                          is_training):
        """ A misconception tower with additional fishing range classiication.

        Args:
            input: a tensor of size [batch_size, 1, width, depth].
            window_size: the width of the conv and pooling filters to apply.
            stride: the downsampling to apply when filtering.
            depth: the depth of the output tensor.
            levels: The height of the tower in misconception layers.

        Returns:
            a tensor of size [batch_size, num_classes].
        """
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
            net = input
            # Three levels of misconception w/out narrowing for fishing prediction.
            net = slim.repeat(net, 3, misconception_with_bypass,
                              self.window_size, 1, self.feature_depth,
                              is_training)

            fishing_prediction = net

            # Then a tower for classification.
            net = slim.repeat(net, self.levels, misconception_with_bypass,
                              self.window_size, self.stride,
                              self.feature_depth, is_training)
            net = slim.flatten(net)
            net = slim.dropout(net, 0.5, is_training=is_training)
            net = slim.fully_connected(net, 100)
            net = slim.dropout(net, 0.5, is_training=is_training)

            logits = []
            for of in objective_functions:
                # TODO(alexwilson): The objective function should have a
                # function to build this last layer.
                logits.append(slim.fully_connected(net, of.num_classes))

            # TODO(alexwilson): Should the last layer have a sigmoid activation
            # function?
            logits.append(fishing_prediction)

            return logits

    def zero_pad_features(self, features):
        """ Zero-pad features in the depth dimension to match requested feature depth. """

        feature_pad_size = self.feature_depth - self.num_feature_dimensions
        assert (feature_pad_size >= 0)
        zero_padding = tf.zeros(
            [self.batch_size, 1, self.window_max_points, feature_pad_size])
        padded = tf.concat(3, [features, zero_padding])

        return padded

    def build_training_net(self, features, labels, fishing_timeseries_labels):

        features = self.zero_pad_features(features)
        one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)

        vessel_class_logits, fishing_prediction_logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.num_classes, True)

        vessel_classification_loss = slim.losses.softmax_cross_entropy(
            vessel_class_logits, one_hot_labels)

        fishing_timeseries_loss = utility.fishing_localisation_loss(
            fishing_prediction_logits, fishing_timeseries_labels)

        total_loss = vessel_classification_loss + fishing_timeseries_loss

        optimizer = tf.train.AdamOptimizer(2e-5)

        return TrainNetInfo(total_loss, optimizer, vessel_class_logits,
                            fishing_prediction_logits)

    def build_inference_net(self, features):

        features = self.zero_pad_features(features)

        vessel_class_logits, fishing_prediction_logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.num_classes, False)

        return vessel_class_logits, fishing_prediction_logits
