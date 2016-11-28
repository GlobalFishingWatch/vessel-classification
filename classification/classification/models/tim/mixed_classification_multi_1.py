from __future__ import print_function, division
import tensorflow as tf
from classification import utility
from collections import namedtuple
import tensorflow.contrib.slim as slim
import logging
import numpy as np

from classification.model import ModelBase

from classification.objectives import (SummaryObjective,
    TrainNetInfo, VesselMetadataClassificationObjective, RegressionObjective,
    MultiClassificationObjective, FishingLocalizationObjectiveCrossEntropy)

from .tf_layers import conv1d_layer, dense_layer, misconception_layer, dropout_layer
from .tf_layers import batch_norm

TowerParams = namedtuple("TowerParams",
                         ["filter_count", "filter_widths", "pool_size",
                          "pool_stride", "keep_prob", "shunt"])


class Model(ModelBase):

    initial_learning_rate = 0.01
    learning_decay_rate = 0.99
    decay_examples = 10000
    momentum = 0.9

    fishing_dense_layer = 128

    tower_params = [
        TowerParams(*x)
        for x in [(32, [3], 2, 2, 1.0, True)] * 10 + [(32, [2], 2, 2, 0.8, True
                                                       )]
    ]

    def __init__(self, num_feature_dimensions, vessel_metadata):
        super(self.__class__, self).__init__(num_feature_dimensions,
                                             vessel_metadata)

        def length_or_none(mmsi):
            length = vessel_metadata.vessel_label('length', mmsi)
            if length == '':
                return None

            return np.float32(length)

        self.classification_objective = MultiClassificationObjective(
            "Multiclass", "Vessel detailed class", vessel_metadata)

        self.length_objective = RegressionObjective(
            'length',
            'Vessel length regression',
            length_or_none,
            loss_weight=0.1)

        self.fishing_localisation_objective = FishingLocalizationObjectiveCrossEntropy(
            'fishing_localisation',
            'Fishing localisation',
            vessel_metadata,
            loss_weight=100)

        self.summary_objective = SummaryObjective(
                'histograms', 'Histograms')

        self.objectives = [self.classification_objective,
                           self.length_objective,
                           self.fishing_localisation_objective,
                           self.summary_objective]

    @property
    def max_window_duration_seconds(self):
        return 90 * 24 * 3600

    @property
    def window_max_points(self):
        length = 1
        for tp in reversed(self.tower_params):
            length = length * tp.pool_stride + (tp.pool_size - tp.pool_stride)
        return length

    def build_stack(self, current, is_training, tower_params):

        stack = [current]

        for i, tp in enumerate(tower_params):
            with tf.variable_scope('tower-segment-{}'.format(i + 1)):

                # Misconception stack
                mc = current

                for j, w in enumerate(tp.filter_widths):
                    H, W, C = [int(x) for x in mc.get_shape().dims[1:]]
                    mc = misconception_layer(
                        mc,
                        tp.filter_count,
                        is_training,
                        filter_size=w,
                        padding="SAME",
                        name='misconception-{}'.format(j))

                if tp.shunt:
                    # Build a shunt layer (resnet) to help convergence
                    with tf.variable_scope('shunt'):
                        # Trim current before making the skip layer so that it matches the dimensons of
                        # the mc stack
                        shunt = tf.nn.elu(
                            batch_norm(
                                conv1d_layer(current, 1, tp.filter_count),
                                is_training))
                    current = shunt + mc
                else:
                    current = mc

                stack.append(current)

                current = tf.nn.max_pool(
                    current, [1, 1, tp.pool_size, 1],
                    [1, 1, tp.pool_stride, 1],
                    padding="VALID")
                if tp.keep_prob < 1:
                    current = dropout_layer(current, is_training, tp.keep_prob)

        # Remove extra dimensions
        H, W, C = [int(x) for x in current.get_shape().dims[1:]]
        output = tf.reshape(current, (-1, C))

        return output, stack

    def build_model(self, is_training, current):

        self.summary_objective.build(current)

        # Build a tower consisting of stacks of misconception layers in parallel
        # with size 1 convolutional shortcuts to help train.

        with tf.variable_scope('classification-tower'):
            classification_output, _ = self.build_stack(current, is_training,
                                                        self.tower_params)
            self.classification_objective.build(classification_output)

        with tf.variable_scope('length-tower'):
            length_output, _ = self.build_stack(current, is_training,
                                                self.tower_params)
            self.length_objective.build(length_output)

        with tf.variable_scope('localization-tower'):
            _, localization_layers = self.build_stack(current, is_training,
                                                      self.tower_params)

        # Assemble the fishing score logits
        fishing_sublayers = []
        for l in reversed(localization_layers):
            H, W, C = [int(x) for x in l.get_shape().dims[1:]]
            assert self.window_max_points % W == 0
            # Use repeat + tile + reshape to achieve same effect a np.repeat
            l = tf.reshape(l, (-1, 1, W, 1, C))
            l = tf.tile(l, [1, 1, 1, self.window_max_points // W, 1])
            l = tf.reshape(l, [-1, 1, self.window_max_points, C])
            fishing_sublayers.append(l)
        current = tf.concat(3, fishing_sublayers)
        current = tf.nn.elu(
            batch_norm(
                conv1d_layer(
                    current, 1, self.fishing_dense_layer, name="fishing1"),
                is_training))
        current = conv1d_layer(current, 1, 1, name="fishing_logits")
        fishing_outputs = tf.reshape(current, (-1, self.window_max_points))

        self.fishing_localisation_objective.build(fishing_outputs)

    def build_inference_net(self, features, timestamps, mmsis):

        self.build_model(tf.constant(False), features)

        evaluations = []
        for obj in self.objectives:
            evaluations.append(obj.build_evaluation(timestamps, mmsis))

        return evaluations

    def build_training_net(self, features, timestamps, mmsis):

        self.build_model(tf.constant(True), features)

        trainers = []
        for obj in self.objectives:
            trainers.append(obj.build_trainer(timestamps, mmsis))

        example = slim.get_or_create_global_step() * self.batch_size

        learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, example, self.decay_examples,
            self.learning_decay_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate, self.momentum)

        return TrainNetInfo(optimizer, trainers)
