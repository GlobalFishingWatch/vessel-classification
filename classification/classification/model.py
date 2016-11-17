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
    def batch_size(self):
        return 64

    @property
    def max_window_duration_seconds(self):
        return None

    # We often allocate a much smaller buffer than would fit the specified time
    # sampled at 5 mins intervals, on the basis that the sample is almost
    # always much more sparse.
    @property
    def window_max_points(self):
        return None

    @property
    def min_viable_timeslice_length(self):
        return self.window_max_points / 4

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
        random_state = np.random.RandomState()
        training_mmsis = self.vessel_metadata.weighted_training_list(
            random_state, split, self.max_replication_factor)
        return [
            '%s/%d.tfrecord' % (base_feature_path, mmsi)
            for mmsi in training_mmsis
        ]

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
