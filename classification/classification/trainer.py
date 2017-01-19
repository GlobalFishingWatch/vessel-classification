from __future__ import absolute_import
import argparse
import json
import logging
import math
import numpy as np
import os
from . import evaluation_loop
from . import utility

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics

from tensorflow.python.framework import errors


class Trainer:
    """ Handles the mechanics of training and evaluating a vessel behaviour
        model.
    """

    num_parallel_readers = 32

    # TODO:  Pass in training verbosity flag
    def __init__(self, model, base_feature_path, train_scratch_path):
        self.model = model
        self.training_objectives = model.training_objectives
        self.base_feature_path = base_feature_path
        self.train_scratch_path = train_scratch_path
        self.checkpoint_dir = self.train_scratch_path + '/train'
        self.eval_dir = self.train_scratch_path + '/eval'

    def _feature_data_reader(self, split, is_training):
        """ Concurrent feature data reader.

        For a given data split (Training/Test) and a set of input files that
        comes in via the vessel metadata, repeatedly read from these in
        shuffled order, outputing batches of randomly sampled segments of vessel
        tracks for model training or evaluation. Multiple readers are started
        concurrently, and the multiple samples can be output per vessel depending
        upon the weight set for each (used for generating more samples for vessel
        types for which we have fewer examples).

        Args:
            split: The subset of data to read (Training/Test).
            is_training: whether the data is for training (or evaluation).

        Returns:
            A tuple of tensors:
                1. A tensor of features of dimension [batch_size, 1, width, depth].
                2. A tensor of timestamps, one per feature of dimension [batch_size, width].
                3. A tensor of time bounds for the feature data slices of dimension [batch_size, 2].
                4. A tensor of mmsis for the features, of dimesion [batch_size].

        """
        input_files = self.model.build_training_file_list(
            self.base_feature_path, split)
        filename_queue = tf.train.input_producer(input_files, shuffle=True)
        capacity = 1000
        min_size_after_deque = capacity - self.model.batch_size * 4

        readers = []
        for _ in range(self.num_parallel_readers):
            readers.append(
                utility.random_feature_cropping_file_reader(
                    self.model.vessel_metadata, filename_queue,
                    self.model.num_feature_dimensions + 1, self.model.
                    max_window_duration_seconds, self.model.window_max_points,
                    self.model.min_viable_timeslice_length,
                    self.model.use_ranges_for_training))

        (features, timestamps, time_bounds,
         mmsis) = tf.train.shuffle_batch_join(
             readers,
             self.model.batch_size,
             capacity,
             min_size_after_deque,
             enqueue_many=True,
             shapes=[[
                 1, self.model.window_max_points,
                 self.model.num_feature_dimensions
             ], [self.model.window_max_points], [2], []])

        return features, timestamps, time_bounds, mmsis

    def _make_saver(self):
        # TODO(alexwilson): The saver in 0.11.0rc2 is broken. Remove when
        # Cloud ML advances from 0.11.0rc2.
        if tf.__version__ == '0.11.0rc2':
            return tf.train.Saver(
                variables.get_variables_to_restore(),
                write_version=tf.train.SaverDef.V1)
        else:
            return tf.train.Saver(variables.get_variables_to_restore())

    def run_training(self, master, is_chief):
        """ The function for running a training replica on a worker. """

        features, timestamps, time_bounds, mmsis = self._feature_data_reader(
            utility.TRAINING_SPLIT, True)

        with tf.device("/gpu:0"):
            (optimizer, objectives) = self.model.build_training_net(
                features, timestamps, mmsis)

            loss = tf.reduce_sum(
                [o.loss for o in objectives], reduction_indices=[0])

            train_op = slim.learning.create_train_op(
                loss,
                optimizer,
                update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

            logging.info("Starting slim training loop.")
            session_config = tf.ConfigProto(allow_soft_placement=True)
            slim.learning.train(
                train_op,
                self.checkpoint_dir,
                master=master,
                is_chief=is_chief,
                number_of_steps=200000,
                save_summaries_secs=30,
                save_interval_secs=60,
                saver=self._make_saver(),
                session_config=session_config)

    def run_evaluation(self, master):
        """ The function for running model evaluation on the master. """

        features, timestamps, time_bounds, mmsis = self._feature_data_reader(
            utility.TEST_SPLIT, False)

        objectives = self.model.build_inference_net(features, timestamps,
                                                    mmsis)

        aggregate_metric_maps = [o.build_test_metrics() for o in objectives]

        summary_ops = []
        update_ops = []
        for names_to_values, names_to_updates in aggregate_metric_maps:
            for metric_name, metric_value in names_to_values.iteritems():
                op = tf.summary.scalar(metric_name, metric_value)
                op = tf.Print(op, [metric_value], metric_name)
                summary_ops.append(op)
            for update_op in names_to_updates.values():
                update_ops.append(update_op)

        num_examples = 4096
        num_evals = math.ceil(num_examples / float(self.model.batch_size))

        # Setup the global step.
        slim.get_or_create_global_step()

        merged_summary_ops = tf.summary.merge(summary_ops)
        evaluation_loop.evaluation_loop(
            master,
            self.checkpoint_dir,
            self.eval_dir,
            num_evals=num_evals,
            eval_op=update_ops,
            summary_op=merged_summary_ops,
            eval_interval_secs=120,
            timeout=20 * 60,
            saver=self._make_saver())
