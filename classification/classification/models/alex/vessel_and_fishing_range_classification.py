from __future__ import absolute_import
import argparse
import json
from . import layers
from classification import utility
from classification.model import ModelBase, TrainNetInfo, make_vessel_label_objective, FishingLocalisationObjective
import logging
import math
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Model(ModelBase):

    window_size = 3
    stride = 2
    feature_depth = 50
    levels = 10

    fishing_vessel_type_embedding_depth = 8

    def __init__(self, num_feature_dimensions, vessel_metadata):
        super(self.__class__, self).__init__(num_feature_dimensions,
                                             vessel_metadata)

        self.classification_training_objectives = [
            make_vessel_label_objective(vessel_metadata, 'is_fishing',
                                        'Fishing', ['Fishing', 'Non-fishing']),
            make_vessel_label_objective(
                vessel_metadata, 'label', 'Vessel class',
                utility.VESSEL_CLASS_NAMES), make_vessel_label_objective(
                    vessel_metadata, 'sublabel', 'Vessel detailed class',
                    utility.VESSEL_CLASS_DETAILED_NAMES),
            make_vessel_label_objective(
                vessel_metadata,
                'length',
                'Vessel length',
                utility.VESSEL_LENGTH_CLASSES,
                transformer=utility.vessel_categorical_length_transformer)
        ]

        self.fishing_localisation_objective = FishingLocalisationObjective(
            'fishing_localisation', 'Fishing localisation', vessel_metadata)

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
            # Three levels of misconception w/out narrowing for fishing prediction.
            net = slim.repeat(net, 3, layers.misconception_with_bypass,
                              self.window_size, 1, self.feature_depth,
                              is_training)

            fishing_prediction_layer = net

            # Then a tower for classification.
            net = slim.repeat(
                net, self.levels, layers.misconception_with_bypass,
                self.window_size, self.stride, self.feature_depth, is_training)
            net = slim.flatten(net)
            net = slim.dropout(net, 0.5, is_training=is_training)
            net = slim.fully_connected(net, 100)
            net = slim.dropout(net, 0.5, is_training=is_training)

            vessel_class_embedding = slim.fully_connected(
                net, self.fishing_vessel_type_embedding_depth)
            reshaped_embedding = tf.reshape(vessel_class_embedding, [
                self.batch_size, 1, 1, self.fishing_vessel_type_embedding_depth
            ])
            tiled_embedding = tf.tile(reshaped_embedding,
                                      [1, 1, self.window_max_points, 1])

            fishing_prediction_input = tf.concat(
                3, [fishing_prediction_layer, tiled_embedding])
            fishing_prediction = tf.squeeze(
                slim.conv2d(
                    fishing_prediction_input,
                    1, [1, 20],
                    activation_fn=tf.nn.sigmoid),
                squeeze_dims=[1, 3])

            logits = [slim.fully_connected(net, of.num_classes)
                      for of in self.classification_training_objectives]

            return logits, fishing_prediction

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

        logits_list, fishing_prediction = self.misconception_with_fishing_ranges(
            features, mmsis, True)

        trainers = []
        for i in range(len(self.classification_training_objectives)):
            trainers.append(self.classification_training_objectives[i]
                            .build_trainer(logits_list[i], timestamps, mmsis))

        trainers.append(
            self.fishing_localisation_objective.build_trainer(
                fishing_prediction, timestamps, mmsis))

        optimizer = tf.train.AdamOptimizer(1e-5)

        return TrainNetInfo(optimizer, trainers)

    def build_inference_net(self, features, timestamps, mmsis):

        features = self.zero_pad_features(features)

        logits_list, fishing_prediction = self.misconception_with_fishing_ranges(
            features, mmsis, False)

        evaluations = []
        for i in range(len(self.classification_training_objectives)):
            to = self.classification_training_objectives[i]
            logits = logits_list[i]
            evaluations.append(to.build_evaluation(logits))

        evaluations.append(
            self.fishing_localisation_objective.build_evaluation(
                fishing_prediction))

        return evaluations
