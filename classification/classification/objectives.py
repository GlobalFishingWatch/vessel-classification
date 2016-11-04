import abc
import calendar
from collections import namedtuple
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics
import utility

Trainer = namedtuple("Trainer", ["loss", "update_ops"])
TrainNetInfo = namedtuple("TrainNetInfo", ["optimizer", "objective_trainers"])


class ObjectiveBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, metadata_label, name):
        self.metadata_label = metadata_label
        self.name = name

    @abc.abstractmethod
    def build_trainer(self, predictions, timestamps, mmsis, loss_weight):
        pass

    @abc.abstractmethod
    def build_evaluation(self, predictions):
        pass


class EvaluationBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, metadata_label, name):
        self.metadata_label = metadata_label
        self.name = name

    @abc.abstractmethod
    def build_test_metrics(self, mmsis):
        pass

    @abc.abstractmethod
    def build_json_results(self, predictions):
        pass


class FishingLocalisationObjective(ObjectiveBase):
    def __init__(self, metadata_label, name, vessel_metadata, loss_weight=1.0):
        super(self.__class__, self).__init__(metadata_label, name)
        self.fishing_ranges_map = vessel_metadata.fishing_ranges_map
        self.loss_weight = loss_weight

    def build_trainer(self, predictions, timestamps, mmsis):
        update_ops = []

        # Convert fishing range labels to per-point labels.
        def dense_fishing_labels(mmsis_array, timestamps_array):
            dense_labels_list = []
            for mmsi, timestamps in zip(mmsis_array, timestamps_array):
                dense_labels = np.zeros_like(timestamps, dtype=np.float32)
                dense_labels.fill(-1.0)
                mmsi = int(mmsi)
                if mmsi in self.fishing_ranges_map:
                    for (start_time, end_time,
                         is_fishing) in self.fishing_ranges_map[mmsi]:
                        start_range = calendar.timegm(start_time.utctimetuple(
                        ))
                        end_range = calendar.timegm(end_time.utctimetuple())
                        mask = (timestamps >= start_range) & (
                            timestamps < end_range)
                        dense_labels[mask] = is_fishing
                dense_labels_list.append(dense_labels)
            return np.array(dense_labels_list)

        dense_labels = tf.reshape(
            tf.py_func(dense_fishing_labels, [mmsis, timestamps],
                       [tf.float32]),
            shape=tf.shape(predictions))

        # TODO(alexwilson): Add training accuracy.
        raw_loss = utility.fishing_localisation_mse(predictions, dense_labels)

        update_ops.append(
            tf.scalar_summary('%s/Training loss' % self.name, raw_loss))

        loss = self.loss_weight * raw_loss

        return Trainer(loss, update_ops)

    def build_evaluation(self, predictions):
        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name):
                super(self.__class__, self).__init__(metadata_label, name)

            def build_test_metrics(self, mmsis):
                # TODO(alexwilson): Add streaming weighted MSE here.
                return {}, {}

            def build_json_results(self, fishing_probabilities):
                # TODO(alexwilson): Plumb through the timestamps as well,
                # then zip the two to give fishing probability results.
                return {}

        return Evaluation(self.metadata_label, self.name)


class ClassificationObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 label_from_mmsi,
                 classes,
                 transformer=None,
                 loss_weight=1.0):
        super(self.__class__, self).__init__(metadata_label, name)
        self.label_from_mmsi = label_from_mmsi
        self.classes = classes
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.num_classes = len(classes)
        self.transformer = transformer
        self.loss_weight = loss_weight

    def training_label(self, mmsi):
        """ Return the index of this training label, or if it's unset, return
            -1 so the loss function can ignore the example.
        """
        label_value = self.label_from_mmsi(mmsi)
        if self.transformer:
            label_value = self.transformer(label_value)
        if label_value:
            return self.class_indices[label_value]
        else:
            return -1

    def build_trainer(self, logits, timestamps, mmsis):
        def labels_from_mmsis(mmsis_array):
            return np.vectorize(
                self.training_label, otypes=[np.int32])(mmsis_array)

        # Look up the labels for each mmsi.
        labels = tf.reshape(
            tf.py_func(labels_from_mmsis, [mmsis], [tf.int32]),
            shape=tf.shape(mmsis))

        # Labels outside the one-hot num_classes range are just encoded
        # to all-zeros, so the use of -1 for unknown works fine here
        # when combined with a mask below.
        one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)

        # Set the label weights to zero when we don't know the class.
        label_weights = tf.select(
            tf.equal(labels, -1),
            tf.zeros_like(
                labels, dtype=tf.float32),
            tf.ones_like(
                labels, dtype=tf.float32))

        raw_loss = slim.losses.softmax_cross_entropy(
            logits, one_hot_labels, weight=label_weights)
        loss = raw_loss * self.loss_weight
        class_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

        update_ops = []
        update_ops.append(
            tf.scalar_summary('%s/Training loss' % self.name, raw_loss))

        accuracy = slim.metrics.accuracy(
            labels, class_predictions, weights=label_weights)
        update_ops.append(
            tf.scalar_summary('%s/Training accuracy' % self.name, accuracy))

        return Trainer(loss, update_ops)

    def build_evaluation(self, logits):
        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, training_label_lookup,
                         classes, num_classes, logits):
                super(self.__class__, self).__init__(metadata_label, name)
                self.training_label_lookup = training_label_lookup
                self.classes = classes
                self.num_classes = num_classes
                self.prediction = slim.softmax(logits)

            def build_test_metrics(self, mmsis):
                def labels_from_mmsis(mmsis_array):
                    return np.vectorize(
                        self.training_label_lookup,
                        otypes=[np.int32])(mmsis_array)

                predictions = tf.cast(tf.argmax(self.prediction, 1), tf.int32)

                # Look up the labels for each mmsi.
                labels = tf.reshape(
                    tf.py_func(labels_from_mmsis, [mmsis], [tf.int32]),
                    shape=tf.shape(mmsis))

                label_mask = tf.select(
                    tf.equal(labels, -1), tf.zeros_like(labels),
                    tf.ones_like(labels))

                return metrics.aggregate_metric_map({
                    '%s/Test accuracy' % self.name: metrics.streaming_accuracy(
                        predictions, labels, weights=label_mask),
                })

            def build_json_results(self, class_probabilities):
                max_prob_index = np.argmax(class_probabilities)
                max_probability = float(class_probabilities[max_prob_index])
                max_label = self.classes[max_prob_index]
                full_scores = dict(
                    zip(self.classes, [float(v) for v in class_probabilities]))

                return {
                    'name': self.name,
                    'max_label': max_label,
                    'max_label_probability': max_probability,
                    'label_scores': full_scores
                }

        return Evaluation(self.metadata_label, self.name, self.training_label,
                          self.classes, self.num_classes, logits)


def make_vessel_label_objective(vessel_metadata,
                                label,
                                name,
                                classes,
                                transformer=None):
    return ClassificationObjective(
        label,
        name,
        lambda mmsi: vessel_metadata.vessel_label(label, mmsi),
        classes,
        transformer=transformer)
