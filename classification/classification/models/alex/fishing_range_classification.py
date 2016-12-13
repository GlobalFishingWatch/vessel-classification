from __future__ import absolute_import
import argparse
import json
from . import abstract_models
from . import layers
from classification import utility
from classification.objectives import (
    FishingLocalizationObjectiveCrossEntropy, TrainNetInfo)
import logging
import math
import numpy as np
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model(abstract_models.MisconceptionWithFishingRangesModel):

    window_size = 3
    stride = 2
    feature_depth = 50
    levels = 6

    @property
    def max_window_duration_seconds(self):
        # A fixed-length rather than fixed-duration window.
        return 0

    @property
    def window_max_points(self):
        return 512

    def __init__(self, num_feature_dimensions, vessel_metadata, metrics):
        super(Model, self).__init__(num_feature_dimensions, vessel_metadata)

        def length_or_none(mmsi):
            length = vessel_metadata.vessel_label('length', mmsi)
            if length == '':
                return None

            return np.float32(length)

        self.fishing_localisation_objective = FishingLocalizationObjectiveCrossEntropy(
            'fishing_localisation',
            'Fishing localisation',
            vessel_metadata,
            loss_weight=50.0,
            metrics=metrics)

        self.classification_training_objectives = []
        self.training_objectives = [self.fishing_localisation_objective]

    def build_training_file_list(self, base_feature_path, split):
        random_state = np.random.RandomState()
        training_mmsis = self.vessel_metadata.fishing_range_only_list(
            random_state, split, self.max_replication_factor)
        return [
            '%s/%d.tfrecord' % (base_feature_path, mmsi)
            for mmsi in training_mmsis
        ]

    def build_training_net(self, features, timestamps, mmsis):
        features = self.zero_pad_features(features)
        self.misconception_with_fishing_ranges(features, mmsis, True)

        trainers = [
            self.fishing_localisation_objective.build_trainer(timestamps,
                                                              mmsis)
        ]

        optimizer = tf.train.AdamOptimizer()

        return TrainNetInfo(optimizer, trainers)

    def build_inference_net(self, features, timestamps, mmsis):
        features = self.zero_pad_features(features)
        self.misconception_with_fishing_ranges(features, mmsis, False)

        evaluations = [
            self.fishing_localisation_objective.build_evaluation(timestamps,
                                                                 mmsis)
        ]

        return evaluations
