from __future__ import print_function, division
import tensorflow as tf
from classification import utility
from collections import namedtuple
import tensorflow.contrib.slim as slim
import logging

from classification.model import (ModelBase, TrainNetInfo,
                                  make_vessel_label_objective,
                                  FishingLocalizationObjectiveCrossEntropy)

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

    fishing_per_layer = 16
    fishing_dense_layer = 128

    tower_params = [
        TowerParams(*x)
        for x in [(32, [3], 2, 2, 1.0, True)] * 9 + [(32, [3], 2, 2, 0.8, True)
                                                     ]
    ]

    def __init__(self, num_feature_dimensions, vessel_metadata):
        super(self.__class__, self).__init__(num_feature_dimensions,
                                             vessel_metadata)

        # TODO(bitsofbits): consider moving these to cached properties instead so we don't need init
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

        self.fishing_localisation_objective = FishingLocalizationObjectiveCrossEntropy(
            'fishing_localisation', 'Fishing localisation', vessel_metadata)

        self.training_objectives = self.classification_training_objectives + [
            self.fishing_localisation_objective
        ]

    @property
    def window_max_points(self):
        length = 1
        for tp in reversed(self.tower_params):
            length = length * tp.pool_stride + (tp.pool_size - tp.pool_stride)
        return length

    def build_model(self, is_training, current):

        # Build a tower consisting of stacks of misconception layers in parallel
        # with size 1 convolutional shortcuts to help train.

        layers = []

        for i, tp in enumerate(self.tower_params):
            with tf.variable_scope('tower-segment-{}'.format(i + 1)):

                # Misconception stack
                mc = current
                for j, w in enumerate(tp.filter_widths):
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

                layers.append(current)

                current = tf.nn.max_pool(
                    current, [1, 1, tp.pool_size, 1],
                    [1, 1, tp.pool_stride, 1],
                    padding="VALID")
                if tp.keep_prob < 1:
                    current = dropout_layer(current, is_training, tp.keep_prob)

        # Remove extra dimensions
        H, W, C = [int(x) for x in current.get_shape().dims[1:]]
        current = tf.reshape(current, (-1, C))

        # Determine classification logits
        logit_list = []
        for cto in self.classification_training_objectives:
            with tf.variable_scope("prediction-layer-{}".format(
                    cto.name.replace(' ', '-'))):
                logit_list.append(dense_layer(current, cto.num_classes))

        # Assemble the fishing score logits

        fishing_sublayers = []
        for l in reversed(layers):
            l = tf.slice(l, [0, 0, 0, 0], [-1, -1, -1, self.fishing_per_layer])
            H, W, C = [int(x) for x in l.get_shape().dims[1:]]
            assert self.window_max_points % W == 0
            logging.debug("SUBLAYERS %s %s %s", H, W, C)
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
        fishing_logits = tf.reshape(current, (-1, self.window_max_points))

        return logit_list, fishing_logits

    def build_inference_net(self, features, timestamps, mmsis):
        logits_list, fishing_logits = self.build_model(
            tf.constant(False), features)

        evaluations = []
        for i in range(len(self.classification_training_objectives)):
            to = self.classification_training_objectives[i]
            logits = logits_list[i]
            evaluations.append(to.build_evaluation(logits))

        # TODO(bitsofbits): pass logits instead of scores so we have wider choice of objectives
        fishing_scores = tf.sigmoid(fishing_logits, "fishing-scores")
        evaluations.append(
            self.fishing_localisation_objective.build_evaluation(
                fishing_scores))

        return evaluations

    def build_training_net(self, features, timestamps, mmsis):

        logits_list, fishing_logits = self.build_model(
            tf.constant(True), features)

        # logits_list, fishing_scores = self.misconception_with_fishing_ranges(
        #     features, mmsis, True)

        trainers = []
        for i in range(len(self.classification_training_objectives)):
            trainers.append(self.classification_training_objectives[i]
                            .build_trainer(logits_list[i], timestamps, mmsis))

        trainers.append(
            self.fishing_localisation_objective.build_trainer(
                fishing_logits, timestamps, mmsis))

        example = slim.get_or_create_global_step() * self.batch_size

        learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, example, self.decay_examples,
            self.learning_decay_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate, self.momentum)
        # optimizer = tf.train.AdamOptimizer(1e-5)

        return TrainNetInfo(optimizer, trainers)
