import abc
import calendar
from collections import namedtuple
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics
import utility
""" Terminology in the context of objectives.
    
    Net: the raw input to an objective function, an embeddeding that has not
         yet been shaped for the predictive task in hand.
    Logits: the input to a softmax classifier.
    Prediction: the output of an objective function, be it class probabilities
                from a categorical function, or a continuous output vector for
                a regression.
"""

Trainer = namedtuple("Trainer", ["loss", "update_ops"])
TrainNetInfo = namedtuple("TrainNetInfo", ["optimizer", "objective_trainers"])


class ObjectiveBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, metadata_label, name, loss_weight):
        self.metadata_label = metadata_label
        self.name = name
        self.loss_weight = loss_weight
        self.prediction = None

    @abc.abstractmethod
    def build(self, net):
        pass

    @abc.abstractmethod
    def build_trainer(self, timestamps, mmsis):
        pass

    @abc.abstractmethod
    def build_evaluation(self, timestamps, mmsis):
        pass


class EvaluationBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, metadata_label, name, score):
        self.metadata_label = metadata_label
        self.name = name
        self.score = score

    @abc.abstractmethod
    def build_test_metrics(self):
        pass

    @abc.abstractmethod
    def build_json_results(self, prediction, timestamps):
        pass


class RegressionObjective(ObjectiveBase):
    def __init__(self, metadata_label, name, value_from_mmsi, loss_weight=1.0):
        super(self.__class__, self).__init__(metadata_label, name, loss_weight)
        self.value_from_mmsi = value_from_mmsi

    def build(self, net):
        self.prediction = tf.squeeze(
            slim.fully_connected(
                input, 1, activation_fn=None))

    def _expected_and_mask(self, mmsis):
        def impl(mmsis_array):
            expected = []
            mask = []
            for mmsi in mmsis_array:
                e = self.value_from_mmsi(mmsi)
                if e != None:
                    expected.append(e)
                    mask.append(1.0)
                else:
                    expected.append(0.0)
                    mask.append(0.0)
            return (np.array(
                expected, dtype=np.float32), np.array(
                    mask, dtype=np.float32))

        expected, mask = tf.py_func(impl, [mmsis], [tf.float32, tf.float32])

        return expected, mask

    def _masked_mean_error(self, predictions, mmsis):
        expected, mask = self._expected_and_mask(mmsis)
        count = tf.reduce_sum(mask)
        diff = tf.abs(tf.mul(expected - predictions, mask))

        epsilon = 1e-7
        error = tf.reduce_sum(diff) / tf.maximum(count, epsilon)

        return error

    def build_trainer(self, timestamps, mmsis):
        raw_loss = self._masked_mean_error(self.prediction, mmsis)

        update_ops = []
        update_ops.append(
            tf.scalar_summary('%s/Training loss' % self.name, raw_loss))

        loss = raw_loss * self.loss_weight

        return Trainer(loss, update_ops)

    def build_evaluation(self, timestamps, mmsis):
        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, masked_mean_error,
                         predictions):
                super(self.__class__, self).__init__(metadata_label, name)
                self.masked_mean_error = masked_mean_error
                self.prediction = prediction
                self.mmsis = mmsis

            def build_test_metrics(self):
                raw_loss = self.masked_mean_error(self.prediction, self.mmsis)

                return metrics.aggregate_metric_map({
                    '%s/Test error' % self.name:
                    metrics.streaming_mean(raw_loss)
                })

            def build_json_results(self, prediction, timestamps):
                return {'name': self.name, 'value': prediction}

        return Evaluation(self.metadata_label, self.name,
                          self._masked_mean_error, self.prediction)


class ClassificationObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 label_from_mmsi,
                 classes,
                 transformer=None,
                 loss_weight=1.0):
        super(ClassificationObjective, self).__init__(metadata_label, name,
                                                      loss_weight)
        self.label_from_mmsi = label_from_mmsi
        self.classes = classes
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.num_classes = len(classes)
        self.transformer = transformer

    def build(self, net):
        self.logits = slim.fully_connected(net, self.num_classes)
        self.prediction = slim.softmax(self.logits)

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

    def build_trainer(self, timestamps, mmsis):
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
            self.logits, one_hot_labels, weight=label_weights)
        loss = raw_loss * self.loss_weight
        class_predictions = tf.cast(tf.argmax(self.logits, 1), tf.int32)

        update_ops = []
        update_ops.append(
            tf.scalar_summary('%s/Training loss' % self.name, raw_loss))

        accuracy = slim.metrics.accuracy(
            labels, class_predictions, weights=label_weights)
        update_ops.append(
            tf.scalar_summary('%s/Training accuracy' % self.name, accuracy))

        return Trainer(loss, update_ops)

    def build_evaluation(self, timestamps, mmsis):
        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, training_label_lookup,
                         classes, num_classes, prediction):
                super(Evaluation, self).__init__(metadata_label, name)
                self.training_label_lookup = training_label_lookup
                self.classes = classes
                self.num_classes = num_classes
                self.prediction = prediction
                self.mmsis = mmsis
                self.timestamps = timestamps

            def build_test_metrics(self):
                def labels_from_mmsis(mmsis_array):
                    return np.vectorize(
                        self.training_label_lookup,
                        otypes=[np.int32])(mmsis_array)

                predictions = tf.cast(tf.argmax(self.prediction, 1), tf.int32)

                # Look up the labels for each mmsi.
                labels = tf.reshape(
                    tf.py_func(labels_from_mmsis, [self.mmsis], [tf.int32]),
                    shape=tf.shape(mmsis))

                label_mask = tf.select(
                    tf.equal(labels, -1), tf.zeros_like(labels),
                    tf.ones_like(labels))

                return metrics.aggregate_metric_map({
                    '%s/Test accuracy' % self.name: metrics.streaming_accuracy(
                        predictions, labels, weights=label_mask),
                })

            def build_json_results(self, prediction, timestamps):
                max_prob_index = np.argmax(prediction)
                max_probability = float(prediction[max_prob_index])
                max_label = self.classes[max_prob_index]
                full_scores = dict(
                    zip(self.classes, [float(v) for v in prediction]))

                return {
                    'name': self.name,
                    'max_label': max_label,
                    'max_label_probability': max_probability,
                    'label_scores': full_scores
                }

        return Evaluation(self.metadata_label, self.name, self.training_label,
                          self.classes, self.num_classes, self.prediction)


class AbstractFishingLocalizationObjective(ObjectiveBase):
    def __init__(self, metadata_label, name, vessel_metadata, loss_weight=1.0):
        ObjectiveBase.__init__(self, metadata_label, name, loss_weight)
        self.fishing_ranges_map = vessel_metadata.fishing_ranges_map

    def dense_labels(self, template_shape, timestamps, mmsis):
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

        return tf.reshape(
            tf.py_func(dense_fishing_labels, [mmsis, timestamps],
                       [tf.float32]),
            shape=template_shape)

    @abc.abstractmethod
    def loss_function(self, logits, dense_labels):
        loss_function = None
        return loss_function

    def build_objective_function(self, net):
        self.logits = net
        self.prediction = tf.sigmoid(net)

    def build_trainer(self, timestamps, mmsis):
        update_ops = []

        dense_labels = self.dense_labels(
            tf.shape(self.prediction), timestamps, mmsis)

        raw_loss = self.loss_function(dense_labels)
        update_ops.append(
            tf.scalar_summary('%s/Training loss' % self.name, raw_loss))

        loss = raw_loss * self.loss_weight

        return Trainer(loss, update_ops)

    def build_evaluation(self, scores, timestamps, mmsis):
        dense_labels = self.dense_labels

        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, prediction, timestamps,
                         mmsis):
                super(Evaluation, self).__init__(metadata_label, name)
                self.prediction = prediction

            def build_test_metrics(self):
                labels = dense_labels(self.prediction, self.timestamps,
                                      self.mmsis)
                thresholded_prediction = tf.to_int32(self.prediction > 0.5)

                valid = tf.to_int32(tf.not_equal(labels, -1))
                ones = tf.to_int32(tf.equal(labels, 1))
                weights = tf.to_float(valid)

                raw_metrics = {
                    'Test MSE': slim.metrics.streaming_mean_squared_error(
                        self.prediction, tf.to_float(ones), weights=weights),
                    'Test accuracy': slim.metrics.streaming_accuracy(
                        thresholded_prediction, ones, weights=weights),
                    'Test precision': slim.metrics.streaming_precision(
                        thresholded_prediction, ones, weights=weights),
                    'Test recall': slim.metrics.streaming_recall(
                        thresholded_prediction, ones, weights=weights),
                    'Test fishing fraction': slim.metrics.streaming_accuracy(
                        thresholded_prediction, valid, weights=weights)
                }

                return metrics.aggregate_metric_map(
                    {"{}/{}".format(self.name, k): v
                     for (k, v) in raw_metrics.items()})

            def build_json_results(self, prediction, timestamps):
                # TODO(alexwilson): Plumb through the timestamps as well,
                # then zip the two to give fishing probability results.
                return {}

        return Evaluation(self.metadata_label, self.name, scores, timestamps,
                          mmsis)


class FishingLocalisationObjectiveMSE(AbstractFishingLocalizationObjective):
    def loss_function(self, dense_labels):
        return utility.fishing_localisation_mse(self.prediction, dense_labels)


class FishingLocalizationObjectiveCrossEntropy(
        AbstractFishingLocalizationObjective):
    def loss_function(self, dense_labels):
        fishing_mask = tf.to_float(tf.not_equal(dense_labels, -1))
        fishing_targets = tf.to_float(dense_labels > 0.5)
        return (tf.reduce_mean(fishing_mask *
                               tf.nn.sigmoid_cross_entropy_with_logits(
                                   self.logits, fishing_targets)))


class VesselMetadataClassificationObjective(ClassificationObjective):
    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 classes,
                 transformer=None,
                 loss_weight=1.0):
        super(ClassificationObjective, self).__init__(
            metadata_label, name,
            lambda mmsi: vessel_metadata.vessel_label(label, mmsi), classes,
            transformer, loss_weight)
