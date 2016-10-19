from __future__ import absolute_import
import argparse
import json
from . import layers
from classification.model import ModelBase, TrainNetInfo, ClassificationObjective
import logging
import math
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model(ModelBase):

    window_size = 3
    stride = 2
    feature_depth = 20
    levels = 10

    def zero_pad_features(self, features):
        """ Zero-pad features in the depth dimension to match requested feature depth. """

        feature_pad_size = self.feature_depth - self.num_feature_dimensions
        assert (feature_pad_size >= 0)
        zero_padding = tf.zeros(
            [self.batch_size, 1, self.window_max_points, feature_pad_size])
        padded = tf.concat(3, [features, zero_padding])

        return padded

    def build_training_net(self, features, labels, fishing_timeseries):

        features = self.zero_pad_features(features)
        vessel_class_logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.num_classes, True)

        vessel_class_trainer = ClassificationObjective(
            "Vessel class",
            self.num_classes).build_trainer(vessel_class_logits, labels)

        optimizer = tf.train.AdamOptimizer(2e-5)

        return TrainNetInfo(optimizer, [vessel_class_trainer])

    def build_inference_net(self, features):

        features = self.zero_pad_features(features)

        vessel_class_logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.num_classes, False)

        vessel_class_trainer = ClassificationObjective(
            "Vessel class",
            self.num_classes).build_trainer(vessel_class_logits, labels)

        return [vessel_class_trainer]
