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
from .model import ModelBase
from . import layers
from classification import metadata
from .objectives import (
    FishingLocalizationObjectiveCrossEntropy, TrainNetInfo)
from classification.feature_generation import fishing_feature_generation
import logging
import math
import numpy as np
import six
import os

import tensorflow as tf


class Model(ModelBase):

    window_size = 3
    stride = 2
    feature_depths = [48, 64, 96, 128, 192, 256, 384, 512, 768]
    strides = [2] * len(feature_depths)

    initial_learning_rate = 1e-3
    learning_decay_rate = 0.5
    decay_examples = 10000

    window = (256, 1024)

    @property
    def number_of_steps(self):
        return 50000

    @property
    def max_window_duration_seconds(self):
        # A fixed-length rather than fixed-duration window.
        return 0

    @property
    def window_max_points(self):
        return 1024


    @property
    def batch_size(self):
        return 16

    @staticmethod
    def read_metadata(all_available_ids,
                      metadata_file,
                      fishing_ranges,
                      split):
        return metadata.read_vessel_time_weighted_metadata(
            all_available_ids, metadata_file, fishing_ranges,
            split=split)

    def __init__(self, num_feature_dimensions, vessel_metadata, metrics):
        super(Model, self).__init__(num_feature_dimensions, vessel_metadata)

        def length_or_none(id_):
            length = vessel_metadata.vessel_label('length', id_)
            if length == '':
                return None

            return np.float32(length)

        self.fishing_localisation_objective = FishingLocalizationObjectiveCrossEntropy(
            'fishing_localisation',
            'Fishing-localisation',
            vessel_metadata,
            metrics=metrics,
            window=self.window)

        self.objectives = [self.fishing_localisation_objective]
        self.objective_map = {obj.name : obj for obj in self.objectives}


    def build_training_file_list(self, base_feature_path, split):
        random_state = np.random.RandomState()
        training_ids = self.vessel_metadata.fishing_range_only_list(
            random_state, split)
        return [
            '%s/%s.tfrecord' % (base_feature_path, six.ensure_text(id_))
            for id_ in training_ids
        ]

    def _build_net(self, features, timestamps, ids, is_training):
        layers.misconception_fishing(
            features,
            filters_list=self.feature_depths,
            kernel_size=self.window_size,
            strides_list=self.strides,
            objective_function=self.fishing_localisation_objective,
            training=is_training,
            pre_filters=128,
            post_filters=128,
            post_layers=1
            )


    def make_model_fn(self):
        def _model_fn(features, labels, mode, params):
            is_train = (mode == tf.estimator.ModeKeys.TRAIN)
            ids = features['id']
            time_ranges = features['time_ranges']
            timestamps = features['timestamps']
            features = features['features']
            self._build_net(features, timestamps, ids, is_train)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    "id" : ids,
                    "time_ranges": time_ranges,
                    "timestamps" : timestamps,
                    self.fishing_localisation_objective.name : self.fishing_localisation_objective.prediction
                    }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            global_step = tf.train.get_global_step()

            total_loss = self.fishing_localisation_objective.create_loss(labels)

            learning_rate = tf.train.exponential_decay(
                self.initial_learning_rate, global_step, 
                self.decay_examples, self.learning_decay_rate)

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss=total_loss, 
                                                  global_step=global_step)

                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=total_loss, train_op=train_op)

            assert mode == tf.estimator.ModeKeys.EVAL

            eval_metrics = self.fishing_localisation_objective.create_metrics(labels)

            return tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metric_ops=eval_metrics)
        return _model_fn

    def make_estimator(self, checkpoint_dir):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        return  tf.estimator.Estimator(
            config=tf.estimator.RunConfig(
                            model_dir=checkpoint_dir, 
                            save_summary_steps=20,
                            save_checkpoints_secs=300, 
                            keep_checkpoint_max=10,
                            session_config=session_config),
            model_fn=self.make_model_fn(),
            params={
            })   

    def make_input_fn(self, base_feature_path, split, parallelism, prefetch):
        def input_fn():
            return (fishing_feature_generation.input_fn(
                        self.vessel_metadata,
                        self.build_training_file_list(base_feature_path, split),
                        self.num_feature_dimensions + 1,
                        self.max_window_duration_seconds,
                        self.window_max_points,
                        self.min_viable_timeslice_length,
                        parallelism=parallelism)
                .prefetch(prefetch)
                .shuffle(prefetch)
                .batch(self.batch_size)
                )
        return input_fn

    def make_training_input_fn(self, base_feature_path, num_parallel_reads, prefetch=1024):
        return self.make_input_fn(base_feature_path, metadata.TRAINING_SPLIT, num_parallel_reads, prefetch)

    def make_test_input_fn(self, base_feature_path, num_parallel_reads, prefetch=1024):
        return self.make_input_fn(base_feature_path, metadata.TEST_SPLIT, num_parallel_reads, prefetch)

    def make_prediction_input_fn(self, paths, range_info, parallelism):
        start_date, end_date = range_info
        def input_fn():
            return fishing_feature_generation.predict_input_fn(
                            paths,
                            self.num_feature_dimensions + 1,
                            self.window_max_points,
                            start_date,
                            end_date,
                            self.window,
                            parallelism=parallelism
                    ).batch(1)
        return input_fn

