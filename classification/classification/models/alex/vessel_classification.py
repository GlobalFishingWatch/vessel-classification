from __future__ import absolute_import
import argparse
import json
from . import layers
from classification import utility
from classification.model import ModelBase
from classification.objectives import TrainNetInfo, VesselMetadataClassificationObjective, RegressionObjective
import logging
import math
import numpy as np
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model(ModelBase):

    window_size = 3
    stride = 2
    feature_depth = 50
    levels = 10

    def __init__(self, num_feature_dimensions, vessel_metadata):
        super(self.__class__, self).__init__(num_feature_dimensions,
                                             vessel_metadata)

        def length_or_none(mmsi):
            length = vessel_metadata.vessel_label('length', mmsi)
            if length == '':
                return None

            return np.float32(length)

        self.training_objectives = [
            VesselMetadataClassificationObjective('is_fishing', 'Fishing',
                                                  vessel_metadata,
                                                  ['Fishing', 'Non-fishing']),
            VesselMetadataClassificationObjective(
                'label', 'Vessel class', vessel_metadata,
                utility.VESSEL_CLASS_NAMES),
            VesselMetadataClassificationObjective(
                    'sublabel', 'Vessel detailed class',
                    vessel_metadata,
                    utility.VESSEL_CLASS_DETAILED_NAMES),
            VesselMetadataClassificationObjective(
                'length',
                'Vessel length category',
                vessel_metadata,
                utility.VESSEL_LENGTH_CLASSES,
                transformer=utility.vessel_categorical_length_transformer),
            RegressionObjective(
                'length',
                'Vessel length regression',
                length_or_none,
                loss_weight=0.1)
        ]

    def zero_pad_features(self, features):
        """ Zero-pad features in the depth dimension to match requested feature depth. """

        feature_pad_size = self.feature_depth - self.num_feature_dimensions
        assert (feature_pad_size >= 0)
        zero_padding = tf.zeros(
            [self.batch_size, 1, self.window_max_points, feature_pad_size])
        padded = tf.concat(3, [features, zero_padding])

        return padded

    def build_training_net(self, features, timestamps, mmsis):

        features = self.zero_pad_features(features)

        logits_list = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.training_objectives, True)

        trainers = []
        for i in range(len(self.training_objectives)):
            trainers.append(self.training_objectives[i].build_trainer(
                timestamps, mmsis))

        optimizer = tf.train.AdamOptimizer(1e-5)

        return TrainNetInfo(optimizer, trainers)

    def build_inference_net(self, features, timestamps, mmsis):

        features = self.zero_pad_features(features)

        outputs_list = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.training_objectives, False)

        evaluations = []
        for i in range(len(self.training_objectives)):
            to = self.training_objectives[i]
            output = outputs_list[i]
            evaluations.append(to.build_evaluation(timestamps, mmsis))

        return evaluations
