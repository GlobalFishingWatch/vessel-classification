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

import abc
from collections import namedtuple
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics
import utility


class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    @property
    def number_of_steps(self):
        """Number of training examples to use"""
        return 500000

    @property
    def use_ranges_for_training(self):
        """Choose features overlapping with provided ranges during training"""
        return False

    @property
    def batch_size(self):
        return 64

    @property
    def max_window_duration_seconds(self):
        """ Window max duration in seconds. A value of zero indicates that
            we would instead like to choose a fixed-length window. """
        return None

    # We often allocate a much smaller buffer than would fit the specified time
    # sampled at 5 mins intervals, on the basis that the sample is almost
    # always much more sparse.
    @property
    def window_max_points(self):
        return None

    @property
    def min_viable_timeslice_length(self):
        return 500

    @property
    def max_replication_factor(self):
        return 100.0

    def __init__(self, num_feature_dimensions, vessel_metadata):
        self.num_feature_dimensions = num_feature_dimensions
        if vessel_metadata:
            self.vessel_metadata = vessel_metadata
            self.fishing_ranges_map = vessel_metadata.fishing_ranges_map
        else:
            self.vessel_metadata = None
            self.fishing_ranges_map = None
        self.training_objectives = None

    def build_training_file_list(self, base_feature_path, split):
        boundary = 1 if (split == utility.TRAINING_SPLIT) else self.batch_size
        random_state = np.random.RandomState()
        training_mmsis = self.vessel_metadata.weighted_training_list(
            random_state,
            split,
            self.max_replication_factor,
            boundary=boundary)
        return [
            '%s/%s.tfrecord' % (base_feature_path, mmsi)
            for mmsi in training_mmsis
        ]

    @staticmethod
    def read_metadata(all_available_mmsis,
                      metadata_file,
                      fishing_ranges,
                      fishing_upweight=1.0):
        return utility.read_vessel_multiclass_metadata(
            all_available_mmsis, metadata_file, fishing_ranges,
            fishing_upweight)

    @abc.abstractmethod
    def build_training_net(self, features, timestamps, mmsis):
        """Build net suitable for training model

        Args:
            features : features to feed into net
            timestamps: a list of timestamps, one for each feature point.
            mmsis: a list of mmsis, one for each batch element.

        Returns:
            TrainNetInfo

        """
        optimizer = trainers = None
        return optimizer, trainers

    @abc.abstractmethod
    def build_inference_net(self, features, timestamps, mmsis):
        """Build net suitable for running inference on model

        Args:
            features : features to feed into net
            timestamps: a list of timestamps, one for each feature point.
            mmsis: a list of mmsis, one for each batch element.

        Returns:
            A list of objects derived from EvaluationBase providing
            functionality to log evaluation statistics as well as to
            return the results of inference as JSON.

        """
        return []
