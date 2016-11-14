from __future__ import absolute_import
import argparse
import json
from . import layers
from classification import utility
from classification.model import ModelBase
from classification.objectives import (FishingLocalizationObjectiveCrossEntropy,
                                       RegressionObjective, TrainNetInfo,
                                       VesselMetadataClassificationObjective)
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
    feature_depth = 80
    levels = 9
    batch_size = 64

    def __init__(self, num_feature_dimensions, vessel_metadata):
        super(self.__class__, self).__init__(num_feature_dimensions,
                                             vessel_metadata)

        def length_or_none(mmsi):
            length = vessel_metadata.vessel_label('length', mmsi)
            if length == '':
                return None

            return np.float32(length)

        self._classification_training_objectives = [
            VesselMetadataClassificationObjective('is_fishing', 'Fishing',
                                                  vessel_metadata,
                                                  ['Fishing', 'Non-fishing']),
            VesselMetadataClassificationObjective('label', 'Vessel class',
                                                  vessel_metadata,
                                                  utility.VESSEL_CLASS_NAMES),
            VesselMetadataClassificationObjective(
                'sublabel', 'Vessel detailed class', vessel_metadata,
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
        self.classification_training_objectives = []

        self.fishing_localisation_objective = FishingLocalizationObjectiveCrossEntropy(
            'fishing_localisation',
            'Fishing localisation',
            vessel_metadata,
            loss_weight=50.0)

        self.training_objectives = self.classification_training_objectives + [
            self.fishing_localisation_objective
        ]

    def misconception_with_fishing_ranges(self, input, mmsis, is_training):
        """ A misconception tower with additional fishing range classification.

        Args:
            input: a tensor of size [batch_size, 1, width, depth].
            window_size: the width of the conv and pooling filters to apply.
            stride: the downsampling to apply when filtering.
            depth: the depth of the output tensor.
            levels: The height of the tower in misconception layers.

        Returns:
            a tensor of size [batch_size, num_classes].
        """
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
            net = input

            # Then a tower for classification.
            multiscale_layers = []
            for i in range(self.levels):
                with tf.variable_scope("layer_%d" % i):
                    ddp = net
                    for j in range(i):
                        ddp = utility.duplicate_double_pad(ddp)
                    multiscale_layers.append(ddp)

                    net = layers.misconception_with_bypass(
                        net, self.window_size, self.stride, self.feature_depth,
                        is_training)

            net = slim.flatten(net)
            net = slim.dropout(net, 0.5, is_training=is_training)
            net = slim.fully_connected(net, 100)
            net = slim.dropout(net, 0.5, is_training=is_training)

            concatenated_multiscale_embedding = tf.concat(3, multiscale_layers)

            fishing_outputs = tf.squeeze(
                slim.conv2d(
                    concatenated_multiscale_embedding, 1, [1, 1],
                    activation_fn=None),
                squeeze_dims=[1, 3])

            for of in self.classification_training_objectives:
                of.build(net)

            self.fishing_localisation_objective.build(fishing_outputs)

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

        self.misconception_with_fishing_ranges(features, mmsis, True)

        trainers = []
        for i in range(len(self.classification_training_objectives)):
            trainers.append(self.classification_training_objectives[i]
                            .build_trainer(timestamps, mmsis))

        trainers.append(
            self.fishing_localisation_objective.build_trainer(timestamps,
                                                              mmsis))

        optimizer = tf.train.AdamOptimizer(1e-4)

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
