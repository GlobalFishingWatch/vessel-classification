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

    initial_learning_rate = 1e-4
    learning_decay_rate = 0.5
    decay_examples = 10000

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
            'Fishing-localisation',
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

        example = slim.get_or_create_global_step() * self.batch_size

        learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, example, self.decay_examples,
            self.learning_decay_rate)

        # op = tf.summary.scalar(metric_name, metric_value)


        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        return TrainNetInfo(optimizer, trainers)

    def build_inference_net(self, features, timestamps, mmsis):
        features = self.zero_pad_features(features)
        self.misconception_with_fishing_ranges(features, mmsis, False)

        evaluations = [
            self.fishing_localisation_objective.build_evaluation(timestamps,
                                                                 mmsis)
        ]

        return evaluations


    # TODO(bitsofbits): there is an implicit assumption that a given model
    #  will only call one of build_inference_net or build_training_net, and only once.
    #  We should enforce that or get rid of the restriction (will require some restructuring)

    def export(self, feature_size, last_checkpoint, output_dir):
        """Builds a prediction graph and xports the model.
        Args:
          last_checkpoint: Path to the latest checkpoint file from training.
          output_dir: Path to the folder to be used to output the model.
        """
        logging.info('Exporting prediction graph to %s', output_dir)
        with tf.Session(graph=tf.Graph()) as sess:
            # Build and save prediction meta graph and trained variable values.
            # TODO create placeholders for mmsi, timestamps, features,
            # TODO (amy): I think we could reshape things here if you want to come in with different shapes
            # (for instance single value for MMSI)
            # or (512x12 shape for features)

            features = tf.placeholder(tf.float32, shape=(1, 1, self.window_max_points, feature_size))
            mmsis = tf.placeholder(tf.int32, shape=(1,))
            timestamps = tf.placeholder(tf.int32, shape=(1, self.window_max_points))

            oput_mmsis = tf.identity(mmsis, name='oputmmsis')
            oput_timestamps = tf.identity(timestamps, name='oputtimestamps')

            # Add inputs to net to `inputs` collections to support CloudML prediction.
            inputs = {'timestamps': timestamps.name, 'mmsis': mmsis.name, 'features': features.name}
            tf.add_to_collection('inputs', json.dumps(inputs))

            self.build_inference_net(features, timestamps, mmsis)

            # Add outputs to net to 'outputs' to support CloudML prediction
            outputs = {'mmsis': oput_mmsis.name, 'timestamps': oput_timestamps.name,
                      'fishing_scores': self.fishing_localisation_objective.prediction.name}
            tf.add_to_collection('outputs', json.dumps(outputs))

            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver = tf.train.Saver()
            saver.restore(sess, last_checkpoint)
            saver.export_meta_graph(filename=os.path.join(output_dir, 'export.meta'))
            saver.save(
              sess, os.path.join(output_dir, 'export'), write_meta_graph=False)
