from __future__ import absolute_import
import argparse
import json
from . import layers
import logging
import math
import os
from ... import utility

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


class Trainer(utility.ModelConfiguration):
    """ Handles the mechanics of training and evaluating a vessel behaviour
        model.
    """

    def __init__(self, vessel_metadata, base_feature_path, train_scratch_path):
        utility.ModelConfiguration.__init__(self)

        self.vessel_metadata = vessel_metadata
        self.base_feature_path = base_feature_path
        self.train_scratch_path = train_scratch_path
        self.checkpoint_dir = self.train_scratch_path + '/train'
        self.eval_dir = self.train_scratch_path + '/eval'
        self.num_parallel_readers = 16

    def _feature_files(self, split):
        return ['%s/%d.tfrecord' % (self.base_feature_path, mmsi)
                for mmsi in self.vessel_metadata[split].keys()]

    def _training_data_reader(self, split, is_training):
        """ Concurrent training data reader.

        Given a pattern for a set of input files, repeatedly read from these in
        shuffled order, outputing batches of randomly sampled segments of vessel
        tracks for model training or evaluation. Multiple readers are started
        concurrently, and the multiple samples can be output per vessel depending
        upon the weight set for each (used for generating more samples for vessel
        types for which we have fewer examples).

        Args:
            input_files: a list of input files for training/eval.
            is_training: whether the data is for training (or evaluation).

        Returns:
            A tuple of tensors:
                1. A tensor of features of dimension [batch_size, 1, width, depth]
                2. A tensor of int32 labels of dimension [batch_size]
                3. A tensor of one-hot encodings of the labels of dimension
                    batch_size, num_classes]

        """
        input_files = self._feature_files(split)
        filename_queue = tf.train.input_producer(input_files, shuffle=True)
        capacity = 1000
        min_size_after_deque = capacity - self.batch_size * 4

        max_replication = 8.0

        readers = []
        for _ in range(self.num_parallel_readers):
            readers.append(
                utility.cropping_weight_replicating_feature_file_reader(
                    self.vessel_metadata[split], filename_queue,
                    self.num_feature_dimensions + 1,
                    self.max_window_duration_seconds, self.window_max_points,
                    self.min_viable_timeslice_length, max_replication))

        raw_features, time_bounds, labels = tf.train.shuffle_batch_join(
            readers,
            self.batch_size,
            capacity,
            min_size_after_deque,
            enqueue_many=True,
            shapes=[[1, self.window_max_points, self.num_feature_dimensions],
                    [2], []])

        features = self.zero_pad_features(raw_features)

        one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)

        return features, labels, one_hot_labels

    def run_training(self, master, is_chief):
        """ The function for running a training replica on a worker. """

        features, labels, one_hot_labels = self._training_data_reader(
            'Training', True)

        logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.num_classes, True)

        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

        loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        tf.scalar_summary('Training loss', loss)

        accuracy = slim.metrics.accuracy(labels, predictions)
        tf.scalar_summary('Training accuracy', accuracy)

        optimizer = tf.train.AdamOptimizer(2e-5)
        train_op = slim.learning.create_train_op(
            loss,
            optimizer,
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        model_variables = slim.get_model_variables()

        slim.learning.train(
            train_op,
            self.checkpoint_dir,
            master=master,
            is_chief=is_chief,
            number_of_steps=500000,
            save_summaries_secs=30,
            save_interval_secs=60)

    def run_evaluation(self, master):
        """ The function for running model evaluation on the master. """

        features, labels, one_hot_labels = self._training_data_reader('Test',
                                                                      False)

        logits = layers.misconception_model(
            features, self.window_size, self.stride, self.feature_depth,
            self.levels, self.num_classes, False)

        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

        names_to_values, names_to_updates = metrics.aggregate_metric_map({
            'Test accuracy': metrics.streaming_accuracy(predictions, labels),
            'Test precision': metrics.streaming_precision(predictions, labels),
        })

        # Create the summary ops such that they also print out to std output.
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.scalar_summary(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        num_examples = 256
        num_evals = math.ceil(num_examples / float(self.batch_size))

        # Setup the global step.
        slim.get_or_create_global_step()

        while True:
            try:
                slim.evaluation.evaluation_loop(
                    master,
                    self.checkpoint_dir,
                    self.eval_dir,
                    num_evals=num_evals,
                    eval_op=names_to_updates.values(),
                    summary_op=tf.merge_summary(summary_ops),
                    eval_interval_secs=120)
            except ValueError as e:
                logging.warning('Error in evaluation loop: (%s), retrying',
                                str(e))
