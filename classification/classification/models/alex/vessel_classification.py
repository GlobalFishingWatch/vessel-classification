from __future__ import absolute_import
import argparse
import json
from . import abstract_models
from . import layers
from classification import utility
from classification.objectives import (
    TrainNetInfo, VesselMetadataClassificationObjective, RegressionObjective)
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

    @property
    def max_window_duration_seconds(self):
        return 180 * 24 * 3600

    @property
    def window_max_points(self):
        return (self.max_window_duration_seconds / (5 * 60)) / 4

    def __init__(self, num_feature_dimensions, vessel_metadata):
        super(Model, self).__init__(num_feature_dimensions, vessel_metadata)

        def length_or_none(mmsi):
            length = vessel_metadata.vessel_label('length', mmsi)
            if length == '':
                return None

            return np.float32(length)

        self.training_objectives = [
            VesselMetadataClassificationObjective('is_fishing', 'Fishing',
                                                  vessel_metadata,
                                                  ['Fishing', 'Non-fishing']),
            VesselMetadataClassificationObjective('label', 'Vessel class',
                                                  vessel_metadata,
                                                  utility.VESSEL_CLASS_NAMES),
            VesselMetadataClassificationObjective(
                'sublabel', 'Vessel detailed class', vessel_metadata,
                utility.VESSEL_CLASS_DETAILED_NAMES), RegressionObjective(
                    'length',
                    'Vessel length regression',
                    length_or_none,
                    loss_weight=0.1)
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

        optimizer = tf.train.AdamOptimizer(1e-5)

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
