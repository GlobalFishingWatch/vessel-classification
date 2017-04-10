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

from __future__ import absolute_import, division
import argparse
import json
from . import abstract_models
from . import layers
from classification import utility
from classification.objectives import (
    TrainNetInfo, MultiClassificationObjective, LogRegressionObjective)
import logging
import math
import numpy as np
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model(abstract_models.MisconceptionModel):

    window_size = 3
    feature_depths = [48, 64, 96, 128, 192, 256, 384, 512, 768]
    strides = [2] * 9
    assert len(strides) == len(feature_depths)
    feature_sub_depths = 1024

    initial_learning_rate = 1e-4
    learning_decay_rate = 0.5
    decay_examples = 50000

    @property
    def number_of_steps(self):
        return 600000

    @property
    def max_window_duration_seconds(self):
        return 180 * 24 * 3600

    @property
    def window_max_points(self):
        nominal_max_points = (self.max_window_duration_seconds / (5 * 60)) / 4
        layer_reductions = np.prod(self.strides)
        final_size = int(round(nominal_max_points / layer_reductions))
        max_points = final_size * layer_reductions
        logging.info('Using %s points', max_points)
        return max_points

    @property
    def min_viable_timeslice_length(self):
        return 500

    def __init__(self, num_feature_dimensions, vessel_metadata, metrics):
        super(Model, self).__init__(num_feature_dimensions, vessel_metadata)

        class XOrNone:
            def __init__(self, key):
                self.key = key

            def __call__(self, mmsi):
                x = vessel_metadata.vessel_label(self.key, mmsi)
                if x == '':
                    return None
                return np.float32(x)

        self.training_objectives = [
            LogRegressionObjective(
                'length',
                'Vessel-length',
                XOrNone('length'),
                metrics=metrics),
            LogRegressionObjective(
                'tonnage',
                'Vessel-tonnage',
                XOrNone('tonnage'),
                metrics=metrics),
            LogRegressionObjective(
                'engine_power',
                'Vessel-engine-Power',
                XOrNone('engine_power'),
                metrics=metrics),
            MultiClassificationObjective(
                "Multiclass", "Vessel-class", vessel_metadata, metrics=metrics)
        ]

    def _build_model(self, features, timestamps, mmsis, is_training):
        outputs, _ = layers.misconception_model(
            features,
            self.window_size,
            self.feature_depths,
            self.strides,
            self.training_objectives,
            is_training,
            sub_count=self.feature_sub_depths,
            sub_layers=2)
        return outputs

    def build_training_net(self, features, timestamps, mmsis):
        self._build_model(features, timestamps, mmsis, is_training=True)

        trainers = []
        for i in range(len(self.training_objectives)):
            trainers.append(self.training_objectives[i].build_trainer(
                timestamps, mmsis))

        learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, slim.get_or_create_global_step(), 
            self.decay_examples, self.learning_decay_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        return TrainNetInfo(optimizer, trainers)

    def build_inference_net(self, features, timestamps, mmsis):
        self._build_model(features, timestamps, mmsis, is_training=False)

        evaluations = []
        for i in range(len(self.training_objectives)):
            to = self.training_objectives[i]
            evaluations.append(to.build_evaluation(timestamps, mmsis))

        return evaluations
