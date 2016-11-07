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
    def build_trainer(self, logits, timestamps, mmsis, loss_weight):
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
    def build_test_metrics(self, mmsis, timestamps):
        pass

    @abc.abstractmethod
    def build_json_results(self, predictions):
        pass


class AbstractFishingLocalizationObjective(ObjectiveBase):
    def __init__(self, metadata_label, name, vessel_metadata, loss_weight=1.0):
        ObjectiveBase.__init__(self, metadata_label, name)
        self.fishing_ranges_map = vessel_metadata.fishing_ranges_map
        self.loss_weight = loss_weight

    def dense_labels(self, template, timestamps, mmsis):

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
            shape=tf.shape(template))

    @abc.abstractmethod
    def loss_function(self, logits, dense_labels):
        loss_function = None
        return loss_function

    def build_trainer(self, logits, timestamps, mmsis, loss_weight=1.0):
        update_ops = []

        dense_labels = self.dense_labels(logits, timestamps, mmsis)

        raw_loss = self.loss_function(logits, dense_labels)

        # TODO(alexwilson): Add training accuracy.

        update_ops.append(
            tf.scalar_summary('%s/Training loss' % self.name, raw_loss))

        loss = self.loss_weight * raw_loss

        return Trainer(loss, update_ops)

    def build_evaluation(self, scores):

        dense_labels = self.dense_labels

        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, scores):
                super(Evaluation, self).__init__(metadata_label, name)
                self.scores = scores

            def build_test_metrics(self, mmsis, timestamps):

                labels = dense_labels(self.scores, timestamps, mmsis)
                predictions = tf.to_int32(self.scores > 0.5)

                valid = tf.to_int32(tf.not_equal(labels, -1))
                ones = tf.to_int32(tf.equal(labels, 1))
                weights = tf.to_float(valid)

                raw_metrics = {
                    'Test MSE': slim.metrics.streaming_mean_squared_error(
                        self.scores, tf.to_float(ones), weights=weights),
                    'Test accuracy': slim.metrics.streaming_accuracy(
                        predictions, ones, weights=weights),
                    'Test precision': slim.metrics.streaming_precision(
                        predictions, ones, weights=weights),
                    'Test recall': slim.metrics.streaming_recall(
                        predictions, ones, weights=weights),
                    'Test fishing fraction': slim.metrics.streaming_accuracy(
                        predictions, valid, weights=weights)
                }

                return metrics.aggregate_metric_map(
                    {"{}/{}".format(self.name, k): v
                     for (k, v) in raw_metrics.items()})

            def build_json_results(self, fishing_probabilities):
                # TODO(alexwilson): Plumb through the timestamps as well,
                # then zip the two to give fishing probability results.
                return {}

        return Evaluation(self.metadata_label, self.name, scores)


class FishingLocalisationObjectiveMSE(AbstractFishingLocalizationObjective):
    def loss_function(self, logits, dense_labels):
        predictions = tf.sigmoid(logits)
        return utility.fishing_localisation_mse(predictions, dense_labels)


class FishingLocalizationObjectiveCrossEntropy(
        AbstractFishingLocalizationObjective):
    def loss_function(self, logits, dense_labels):
        fishing_mask = tf.to_float(tf.not_equal(dense_labels, -1))
        fishing_targets = tf.to_float(dense_labels > 0.5)
        return (tf.reduce_mean(fishing_mask *
                               tf.nn.sigmoid_cross_entropy_with_logits(
                                   logits, fishing_targets)))


class ClassificationObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 label_from_mmsi,
                 classes,
                 transformer=None,
                 loss_weight=1.0):
        super(ClassificationObjective, self).__init__(metadata_label, name)
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
                super(Evaluation, self).__init__(metadata_label, name)
                self.training_label_lookup = training_label_lookup
                self.classes = classes
                self.num_classes = num_classes
                self.prediction = slim.softmax(logits)

            def build_test_metrics(self, mmsis, timestamps):
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



            # make_vessel_label_objective(
            #     vessel_metadata, 'label', 'Vessel class',
            #     utility.VESSEL_CLASS_NAMES), make_vessel_label_objective(
            #         vessel_metadata, 'sublabel', 'Vessel detailed class',
            #         utility.VESSEL_CLASS_DETAILED_NAMES),
class MultiClassificationObjective(ObjectiveBase):
    def __init__(self, metadata_label, name, vessel_metadata, loss_weight=1.0):

        super(MultiClassificationObjective, self).__init__(metadata_label, name)
        # self.label_from_mmsi = label_from_mmsi
        self.vessel_metadata = vessel_metadata # Use to with MMSI to get key
        self.loss_weight = loss_weight
        self.classes = utility.VESSEL_CLASS_DETAILED_NAMES
        self.num_classes = utility.multihot_lookup_table.shape[-1]

    def training_label(self, mmsi, label):
        """ Return the index of this training label, or if it's unset, return
            -1 so the loss function can ignore the example.
        """
        return self.vessel_metadata.vessel_label(label, mmsi) or -1




    def build_trainer(self, logits, timestamps, mmsis):
        get_vessel_label = self.vessel_metadata.vessel_label
        def labels_from_mmsis(seq, label, class_indices):
            result = np.empty([len(seq)], dtype=np.int32)
            for i, m in enumerate(seq):
                lbl_str = get_vessel_label(label, m)
                if lbl_str:
                    result[i] = class_indices[lbl_str]
                else:
                    result[i] = -1
            return result


        # Look up the labels for each mmsi.
        fishing_inds = {k: i for (i, k) in enumerate(['Fishing', 'Non-fishing'])}
        is_fishing = tf.reshape(
            tf.py_func(lambda x: labels_from_mmsis(x, 'is_fishing', fishing_inds), [mmsis], [tf.int32]),
            shape=tf.shape(mmsis))

        coarse_inds = {k: i for (i, k) in enumerate(utility.VESSEL_CLASS_NAMES)}
        coarse = tf.reshape(
            tf.py_func(lambda x: labels_from_mmsis(x, 'label', coarse_inds), [mmsis], [tf.int32]),
            shape=tf.shape(mmsis))

        fine_inds = {k: i for (i, k) in enumerate(utility.VESSEL_CLASS_DETAILED_NAMES)}
        fine = tf.reshape(
            tf.py_func(lambda x: labels_from_mmsis(x, 'sublabel', fine_inds), [mmsis], [tf.int32]),
            shape=tf.shape(mmsis))

        multihot_labels = utility.multihot_encode(is_fishing=is_fishing, coarse=coarse, fine=fine)

        # raw_loss = slim.losses.softmax_cross_entropy(
        #     logits, multihot_labels)

        with tf.variable_scope("custom-loss"):
            softmax = tf.nn.softmax(logits)
            total_positives = tf.reduce_sum(tf.to_float(multihot_labels) * softmax, reduction_indices=[1])
            raw_loss = -tf.reduce_mean(tf.log(total_positives))

        loss = raw_loss * self.loss_weight
        class_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

        update_ops = []
        update_ops.append(
            tf.scalar_summary('%s/Training loss' % self.name, raw_loss))

        # accuracy = slim.metrics.accuracy(
        #     labels, class_predictions, weights=label_weights)
        # update_ops.append(
        #     tf.scalar_summary('%s/Training accuracy' % self.name, accuracy))

        return Trainer(loss, update_ops)

    def build_evaluation(self, logits):
        
        _get_vessel_label = self.vessel_metadata.vessel_label


        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, training_label_lookup,
                         classes, num_classes, logits):
                super(Evaluation, self).__init__(metadata_label, name)
                self.training_label_lookup = training_label_lookup
                self.classes = classes
                self.num_classes = num_classes
                self.prediction = slim.softmax(logits)

            def build_test_metrics(self, mmsis, timestamps):
                def labels_from_mmsis(seq, label, class_indices):
                    get_vessel_label = _get_vessel_label
                    result = np.empty([len(seq)], dtype=np.int32)
                    for i, m in enumerate(seq):
                        lbl_str = get_vessel_label(label, m)
                        if lbl_str:
                            result[i] = class_indices[lbl_str]
                        else:
                            result[i] = -1
                    return result


                fine_inds = {k: i for (i, k) in enumerate(utility.VESSEL_CLASS_DETAILED_NAMES)}
                fine_labels = tf.reshape(
                    tf.py_func(lambda x: labels_from_mmsis(x, 'sublabel', fine_inds), [mmsis], [tf.int32]),
                    shape=tf.shape(mmsis))

                fine_predictions = tf.cast(tf.argmax(self.prediction, 1), tf.int32)

                fine_mask = tf.select(
                    tf.equal(fine_labels, -1), tf.zeros_like(fine_labels),
                    tf.ones_like(fine_labels))



                coarse_inds = {k: i for (i, k) in enumerate(utility.VESSEL_CLASS_NAMES)}
                coarse_labels = tf.reshape(
                    tf.py_func(lambda x: labels_from_mmsis(x, 'label', coarse_inds), [mmsis], [tf.int32]),
                    shape=tf.shape(mmsis))


                batch_size = tf.shape(mmsis)[0]
                coarse_lookup = tf.to_float(tf.tile(tf.convert_to_tensor(utility.multihot_coarse_lookup_table[np.newaxis,:,:]), 
                                        [batch_size, 1, 1]))
                raw_coarse_prediction = tf.reshape(
                    tf.batch_matmul(coarse_lookup, 
                        tf.reshape(self.prediction, 
                            [batch_size, len(utility.VESSEL_CLASS_DETAILED_NAMES), 1])),
                                [batch_size, len(utility.VESSEL_CLASS_NAMES)])

                coarse_prediction = tf.cast(tf.argmax(raw_coarse_prediction, 1), tf.int32)


                coarse_mask = tf.select(
                    tf.equal(coarse_labels, -1), tf.zeros_like(coarse_labels),
                    tf.ones_like(coarse_labels))


                fishing_inds = {k: i for (i, k) in enumerate(utility.FISHING_NONFISHING_NAMES)}
                is_fishing = tf.reshape(
                    tf.py_func(lambda x: labels_from_mmsis(x, 'is_fishing', fishing_inds), [mmsis], [tf.int32]),
                    shape=tf.shape(mmsis))

                fishing_mask = tf.select(
                    tf.equal(is_fishing, -1), tf.zeros_like(is_fishing),
                    tf.ones_like(is_fishing))

                batch_size = tf.shape(is_fishing)[0]
                fishing_lookup = tf.to_float(tf.tile(tf.convert_to_tensor(utility.multihot_fishing_lookup_table[np.newaxis,:,:]), 
                                        [batch_size, 1, 1]))
                raw_fishing_prediction = tf.reshape(
                    tf.batch_matmul(fishing_lookup, 
                        tf.reshape(self.prediction, 
                            [batch_size, len(utility.VESSEL_CLASS_DETAILED_NAMES), 1])),
                                [batch_size, 2])

                fishing_prediction = tf.cast(tf.argmax(raw_fishing_prediction, 1), tf.int32)


                # TODO: (bitsofbits) refactor to make not horrible

                return metrics.aggregate_metric_map({
                    '%s/Test fine accuracy' % self.name: metrics.streaming_accuracy(
                        fine_predictions, fine_labels, weights=fine_mask),
                    '%s/Test coarse accuracy' % self.name: metrics.streaming_accuracy(
                        coarse_prediction, coarse_labels, weights=coarse_mask),
                    '%s/Test fishing accuracy' % self.name: metrics.streaming_accuracy(
                        fishing_prediction, is_fishing, weights=fishing_mask),
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






class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    batch_size = 32

    feature_duration_days = 180
    max_sample_frequency_seconds = 5 * 60
    max_window_duration_seconds = feature_duration_days * 24 * 3600

    # We allocate a much smaller buffer than would fit the specified time
    # sampled at 5 mins intervals, on the basis that the sample is almost
    # always much more sparse.
    window_max_points = (max_window_duration_seconds /
                         max_sample_frequency_seconds) / 4

    min_viable_timeslice_length = 500

    def __init__(self, num_feature_dimensions, vessel_metadata):
        self.num_feature_dimensions = num_feature_dimensions
        if vessel_metadata:
            self.vessel_metadata = vessel_metadata
            self.fishing_ranges_map = vessel_metadata.fishing_ranges_map
        else:
            self.vessel_metadata = None
            self.fishing_ranges_map = None
        self.training_objectives = None

    @abc.abstractmethod
    def build_training_net(self, features, timestamps, mmsis):
        """Build net suitable for training model

        Args:
            features : features to feed into net
            timestamps: a list of timestamps, one for each feature point.
            mmsis: a list of mmsis, one for each batch element.

        Returns:
            TrainNetInfo

        """
        optimizer = trainers = None
        return optimizer, trainers

    @abc.abstractmethod
    def build_inference_net(self, features, timestamps, mmsis):
        """Build net suitable for running inference on model

        Args:
            features : features to feed into net
            timestamps: a list of timestamps, one for each feature point.
            mmsis: a list of mmsis, one for each batch element.

        Returns:
            A list of objects derived from EvaluationBase providing
            functionality to log evaluation statistics as well as to
            return the results of inference as JSON.

        """
        return []
