# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
import argparse
import json
from . import abstract_models
from . import layers
from classification import utility
from classification.objectives import (
    FishingLocalizationObjectiveCrossEntropy, RegressionObjective,
    TrainNetInfo, MultiClassificationObjective)
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
    feature_depth = 80
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

        self.classification_training_objectives = [
            MultiClassificationObjective(
                "Multiclass",
                "Vessel-class",
                vessel_metadata,
                metrics=metrics), RegressionObjective(
                    'length',
                    'Vessel length regression',
                    length_or_none,
                    loss_weight=0.1,
                    metrics=metrics)
        ]

        self.fishing_localisation_objective = FishingLocalizationObjectiveCrossEntropy(
            'fishing_localisation',
            'Fishing-localisation',
            vessel_metadata,
            loss_weight=50.0,
            metrics=metrics)

        self.training_objectives = self.classification_training_objectives + [
            self.fishing_localisation_objective
        ]

    def build_training_file_list(self, base_feature_path, split):
        random_state = np.random.RandomState()
        training_mmsis = self.vessel_metadata.weighted_training_list(
            random_state, split, self.max_replication_factor,
            lambda row: row['is_fishing'] == 'Fishing')
        return [
            '%s/%d.tfrecord' % (base_feature_path, mmsi)
            for mmsi in training_mmsis
        ]

    def build_training_net(self, features, timestamps, mmsis):
        features = self.zero_pad_features(features)
        self.misconception_with_fishing_ranges(features, mmsis, True)

        trainers = []
        for i in range(len(self.classification_training_objectives)):
            trainers.append(self.classification_training_objectives[i]
                            .build_trainer(timestamps, mmsis))

        trainers.append(
            self.fishing_localisation_objective.build_trainer(timestamps,
                                                              mmsis))

        optimizer = tf.train.AdamOptimizer()

        return TrainNetInfo(optimizer, trainers)

    def build_inference_net(self, features, timestamps, mmsis):
        features = self.zero_pad_features(features)
        self.misconception_with_fishing_ranges(features, mmsis, False)

        evaluations = []
        for i in range(len(self.classification_training_objectives)):
            to = self.classification_training_objectives[i]
            evaluations.append(to.build_evaluation(timestamps, mmsis))

        evaluations.append(
            self.fishing_localisation_objective.build_evaluation(timestamps,
                                                                 mmsis))

        return evaluations
