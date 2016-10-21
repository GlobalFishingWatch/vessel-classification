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

    def __init__(self, num_feature_dimensions):
        super(self.__class__, self).__init__(num_feature_dimensions)

    def zero_pad_features(self, features):
        """ Zero-pad features in the depth dimension to match requested feature depth. """

        feature_pad_size = self.feature_depth - self.num_feature_dimensions
        assert (feature_pad_size >= 0)
        zero_padding = tf.zeros(
            [self.batch_size, 1, self.window_max_points, feature_pad_size])
        padded = tf.concat(3, [features, zero_padding])

        return padded

    def build_training_net(self, features, label_sets, fishing_timeseries):

        features = self.zero_pad_features(features)

        vessel_class_logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.training_objectives, True)

        trainers = []
        for (logits, to, labels) in self.training_objectives:
            trainers.append(to.build_trainer(logits, labels))

        optimizer = tf.train.AdamOptimizer(2e-5)

        return TrainNetInfo(optimizer, trainers)

    def build_inference_net(self, features):

        features = self.zero_pad_features(features)

        vessel_class_logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.training_objectives, False)

        evaluations = []
        for (logits, to) in self.training_objectives:
            evaluations.append(to.build_evaluation(logits))

        return [evaluations]
