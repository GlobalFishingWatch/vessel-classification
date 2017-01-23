from __future__ import absolute_import, division
import argparse
import json
from . import abstract_models
from . import layers
from classification import utility
from classification.objectives import (
    TrainNetInfo, MultiClassificationObjective, RegressionObjective)
import logging
import math
import numpy as np
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model(abstract_models.MisconceptionModel):

    window_size = 3
    stride = 2
    feature_depth = 80
    levels = 9

    initial_learning_rate = 1e-3
    learning_decay_rate = 0.5
    decay_examples = 20000

    @property
    def max_window_duration_seconds(self):
        return 180 * 24 * 3600

    @property
    def window_max_points(self):
        nominal_max_points = (self.max_window_duration_seconds / (5 * 60)) / 4
        layer_reductions = 2 ** self.levels
        final_size = int(round(nominal_max_points / layer_reductions))
        max_points = final_size * layer_reductions
        logging.info('Using %s points', max_points)
        return max_points

    @property
    def min_viable_timeslice_length(self):
        return 500

    def __init__(self, num_feature_dimensions, vessel_metadata, metrics):
        super(Model, self).__init__(num_feature_dimensions, vessel_metadata)

        self.training_objectives = [
            MultiClassificationObjective(
                "Multiclass",
                "Vessel-class",
                vessel_metadata,
                metrics=metrics)
        ]

    def build_training_net(self, features, timestamps, mmsis):

        features = self.zero_pad_features(features)

        layers.misconception_model(features, self.window_size, self.stride,
                                   self.feature_depth, self.levels,
                                   self.training_objectives, True)

        trainers = []
        for i in range(len(self.training_objectives)):
            trainers.append(self.training_objectives[i].build_trainer(
                timestamps, mmsis))

        step = slim.get_or_create_global_step() 

        learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, step, self.decay_examples,
            self.learning_decay_rate)

        # op = tf.summary.scalar(metric_name, metric_value)


        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        return TrainNetInfo(optimizer, trainers)

    def build_inference_net(self, features, timestamps, mmsis):

        features = self.zero_pad_features(features)

        layers.misconception_model(features, self.window_size, self.stride,
                                   self.feature_depth, self.levels,
                                   self.training_objectives, False)

        evaluations = []
        for i in range(len(self.training_objectives)):
            to = self.training_objectives[i]
            evaluations.append(to.build_evaluation(timestamps, mmsis))

        return evaluations
