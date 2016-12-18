import abc
import calendar
from collections import namedtuple, OrderedDict
import datetime
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


def f1(recall, precision):
    rval, rop = recall
    pval, pop = precision
    f1 = 2.0 / (1.0 / rval + 1.0 / pval)
    return (f1, f1)


class ObjectiveBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, metadata_label, name, loss_weight, metrics):
        """
        args:
            metadata_label:
            name: name of this objective (for metrics)
            loss_weight: weight of this objective (so that can increase decrease relative to other objectives)
            metrics: which metrics to include. Options are currently ['all', 'minimal']

        """
        self.metadata_label = metadata_label
        self.name = name
        self.loss_weight = loss_weight
        self.prediction = None
        self.metrics = metrics

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

    def __init__(self, metadata_label, name, prediction, timestamps, mmsis,
                 metrics):
        self.metadata_label = metadata_label
        self.name = name
        self.prediction = prediction
        self.timestamps = timestamps
        self.mmsis = mmsis
        self.metrics = metrics

    @abc.abstractmethod
    def build_test_metrics(self):
        pass

    @abc.abstractmethod
    def build_json_results(self, prediction, timestamps):
        pass


class SummaryObjective(ObjectiveBase):
    def __init__(self, metadata_label, name, metrics):
        super(SummaryObjective, self).__init__(metadata_label, name, 0.0,
                                               metrics)

    def build(self, net):
        self.inputs = net

    def _build_summary(self):
        #TODO(bitsofbits): pull these names from someplace
        ops = {}
        if self.metrics == 'all':
            for i, name in enumerate(
                ['log_timestampDeltaSeconds', 'log_distanceDeltaMeters',
                 'log_speedMps', 'log_integratedSpeedMps',
                 'cogDeltaDegrees_div_180', 'localTodFeature',
                 'localMonthOfYearFeature',
                 'integratedCogDeltaDegrees_div_180', 'log_distanceToShoreKm',
                 'log_distanceToBoundingAnchorageKm',
                 'log_timeToBoundingAnchorageS']):
                ops[name] = tf.summary.histogram(
                    "input/{}-{}".format(name, i),
                    tf.reshape(self.inputs[:, :, :, i], [-1]),
                    #TODO(bitsofbits): may need not need all of these collection keys
                    collections=[tf.GraphKeys.UPDATE_OPS,
                                 tf.GraphKeys.SUMMARIES])
        return ops

    def build_trainer(self, timestamps, mmsis):
        ops = self._build_summary()
        # We return a constant loss of zero here, so this doesn't effect the training,
        # only adds summaries to the output.
        return Trainer(0, ops.values())

    def build_evaluation(self, timestamps, mmsis):

        build_summary = self._build_summary

        class Evaluation(EvaluationBase):
            def build_test_metrics(self):
                ops = build_summary()

                return {}, ops

            def build_json_results(self, prediction, timestamps):
                return {}

        return Evaluation(self.metadata_label, self.name, None, None, None,
                          self.metrics)


class RegressionObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 value_from_mmsi,
                 loss_weight=1.0,
                 metrics='all'):
        super(RegressionObjective, self).__init__(metadata_label, name,
                                                  loss_weight, metrics)
        self.value_from_mmsi = value_from_mmsi

    def build(self, net):
        self.prediction = tf.squeeze(
            slim.fully_connected(
                net, 1, activation_fn=None))

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
            tf.summary.scalar('%s/Training-loss' % self.name, raw_loss))

        loss = raw_loss * self.loss_weight

        return Trainer(loss, update_ops)

    def build_evaluation(self, timestamps, mmsis):
        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, masked_mean_error,
                         prediction, metrics):
                super(Evaluation, self).__init__(metadata_label, name,
                                                 prediction, timestamps, mmsis,
                                                 metrics)
                self.masked_mean_error = masked_mean_error
                self.mmsis = mmsis

            def build_test_metrics(self):
                raw_loss = self.masked_mean_error(self.prediction, self.mmsis)

                return metrics.aggregate_metric_map({
                    '%s/Test-error' % self.name:
                    metrics.streaming_mean(raw_loss)
                })

            def build_json_results(self, prediction, timestamps):
                return {'name': self.name, 'value': float(prediction)}

        return Evaluation(self.metadata_label, self.name,
                          self._masked_mean_error, self.prediction,
                          self.metrics)


class MultiClassificationObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 loss_weight=1.0,
                 metrics='all'):
        super(MultiClassificationObjective, self).__init__(
            metadata_label, name, loss_weight, metrics)
        self.vessel_metadata = vessel_metadata
        self.classes = utility.VESSEL_CLASS_DETAILED_NAMES
        self.num_classes = utility.multihot_lookup_table.shape[-1]

    def build(self, net):
        self.logits = slim.fully_connected(net, self.num_classes)
        self.prediction = slim.softmax(self.logits)

    def training_label(self, mmsi, label):
        """ Return the index of this training label, or if it's unset, return
            -1 so the loss function can ignore the example.
        """
        return self.vessel_metadata.vessel_label(label, mmsi) or -1

    def derived_labels(self, names, label_name, mmsis):

        def labels_from_mmsis(seq, label, class_indices):
            result = np.empty([len(seq)], dtype=np.int32)
            for i, m in enumerate(seq):
                lbl_str = self.vessel_metadata.vessel_label(label, m)
                if lbl_str:
                    result[i] = class_indices[lbl_str]
                else:
                    result[i] = -1
            return result

        indices = {k: i for (i, k) in enumerate(names) }
        labels = tf.reshape(
            tf.py_func(lambda x: labels_from_mmsis(x, label_name, indices),
                           [mmsis], [tf.int32]),
            shape=tf.shape(mmsis))

        mask = tf.to_float(tf.not_equal(labels, -1))

        return labels, mask


    def build_trainer(self, timestamps, mmsis):

        fine_labels, _  = self.derived_labels(
                                    utility.VESSEL_CLASS_DETAILED_NAMES, 'sublabel', mmsis)

        coarse_labels, _ = self.derived_labels(
                                    OrderedDict(utility.VESSEL_CATEGORIES['coarse']), 'label', mmsis)

        fishing_labels, _ = self.derived_labels(
                                    OrderedDict(utility.VESSEL_CATEGORIES['fishing']), 'is_fishing', mmsis)

        empty_labels = tf.zeros_like(fine_labels) - 1

        multihot_labels = utility.multihot_encode(
            is_fishing=fishing_labels, coarse=coarse_labels, fine=fine_labels)

        with tf.variable_scope("custom-loss"):
            total_positives = tf.reduce_sum(
                tf.to_float(multihot_labels) * self.prediction,
                reduction_indices=[1])
            raw_loss = -tf.reduce_mean(tf.log(total_positives))

        loss = raw_loss * self.loss_weight

        update_ops = []
        update_ops.append(
            tf.summary.scalar('%s/Training-loss' % self.name, raw_loss))

        return Trainer(loss, update_ops)

    def build_evaluation(self, timestamps, mmsis):

        logits = self.logits

        derived_labels = self.derived_labels

        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, training_label_lookup,
                         classes, num_classes, prediction, metrics):
                super(Evaluation, self).__init__(metadata_label, name,
                                                 prediction, timestamps, mmsis,
                                                 metrics)
                self.training_label_lookup = training_label_lookup
                self.classes = classes
                self.num_classes = num_classes
                self.prediction = slim.softmax(logits)




            def _derived_prediction(self, lookup_table):
                batch_size = tf.shape(self.mmsis)[0]
                lookup = tf.to_float(
                    tf.tile(
                        tf.convert_to_tensor(
                            lookup_table[
                                np.newaxis, :, :]), [batch_size, 1, 1]))
                raw_prediction = tf.reshape(
                    tf.batch_matmul(lookup, tf.reshape(
                        self.prediction,
                        [batch_size, len(utility.VESSEL_CLASS_DETAILED_NAMES),
                         1])),
                    [batch_size, len(lookup_table)])

                return raw_prediction, tf.to_int32(tf.argmax(raw_prediction, 1))


            def build_test_metrics(self):

                fine_labels, fine_mask  = derived_labels(
                                            utility.VESSEL_CLASS_DETAILED_NAMES, 'sublabel', self.mmsis)
                fine_predictions = tf.to_int32(tf.argmax(self.prediction, 1))

                coarse_labels, coarse_mask = derived_labels(
                                            OrderedDict(utility.VESSEL_CATEGORIES['coarse']), 'label', self.mmsis)
                raw_coarse_prediction, coarse_prediction = self._derived_prediction(utility.multihot_coarse_lookup_table)

                fishing_labels, fishing_mask = derived_labels(
                                            OrderedDict(utility.VESSEL_CATEGORIES['fishing']), 'is_fishing', self.mmsis)
                _, fishing_prediction = self._derived_prediction(utility.multihot_fishing_lookup_table)

                metrics_map = {
                    '%s/Test-fine-accuracy' % self.name:
                    metrics.streaming_accuracy(
                        fine_predictions, fine_labels, weights=fine_mask),
                    '%s/Test-coarse-accuracy' % self.name:
                    metrics.streaming_accuracy(
                        coarse_prediction, coarse_labels, weights=coarse_mask),
                    '%s/Test-fishing-accuracy' % self.name:
                    metrics.streaming_accuracy(
                        fishing_prediction, fishing_labels, weights=fishing_mask),
                }

                if self.metrics == 'all':
                    for i, cls in enumerate(self.classes):
                        cls_name = cls.replace(' ', '-')
                        trues = tf.to_int32(tf.equal(fine_labels, i))
                        preds = tf.to_int32(tf.equal(fine_predictions, i))
                        recall = metrics.streaming_recall(
                            preds, trues, weights=fine_mask)
                        precision = metrics.streaming_precision(
                            preds, trues, weights=fine_mask)
                        metrics_map["%s/Class-%s-Precision" %
                                    (self.name, cls_name)] = recall
                        metrics_map["%s/Class-%s-Recall" %
                                    (self.name, cls_name)] = precision
                        metrics_map["%s/Class-%s-F1-Score" %
                                    (self.name, cls_name)] = f1(recall, precision)
                        metrics_map["%s/Class-%s-ROC-AUC" %
                                    (self.name, cls_name)] = metrics.streaming_auc(
                                        self.prediction[:, i],
                                        trues,
                                        weights=fine_mask)

                    for i, (cls, fine) in enumerate(utility.VESSEL_CATEGORIES['coarse']):
                        # Also include coarse classes, but only if they are not
                        # already included in the fine classes
                        if cls in self.classes:
                            continue
                        cls_name = cls.replace(' ', '-')
                        trues = tf.to_int32(tf.equal(coarse_labels, i))
                        preds = tf.to_int32(tf.equal(coarse_prediction, i))
                        recall = metrics.streaming_recall(
                            preds, trues, weights=coarse_mask)
                        precision = metrics.streaming_precision(
                            preds, trues, weights=coarse_mask)
                        metrics_map["%s/Class-%s-Precision" %
                                    (self.name, cls_name)] = recall
                        metrics_map["%s/Class-%s-Recall" %
                                    (self.name, cls_name)] = precision
                        metrics_map["%s/Class-%s-F1-Score" %
                                    (self.name, cls_name)] = f1(recall, precision)
                        metrics_map["%s/Class-%s-ROC-AUC" %
                                    (self.name, cls_name)] = metrics.streaming_auc(
                                        raw_coarse_prediction[:, i],
                                        trues,
                                        weights=coarse_mask)

                return metrics.aggregate_metric_map(metrics_map)

            def build_json_results(self, class_probabilities, timestamps):
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
                          self.classes, self.num_classes, logits, self.metrics)


class AbstractFishingLocalizationObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 loss_weight=1.0,
                 metrics='all'):
        ObjectiveBase.__init__(self, metadata_label, name, loss_weight,
                               metrics)
        self.vessel_metadata = vessel_metadata

    def dense_labels(self, template_shape, timestamps, mmsis):
        # Convert fishing range labels to per-point labels.
        def dense_fishing_labels(mmsis_array, timestamps_array):
            dense_labels_list = []
            for mmsi, timestamps in zip(mmsis_array, timestamps_array):
                dense_labels = np.zeros_like(timestamps, dtype=np.float32)
                dense_labels.fill(-1.0)
                mmsi = int(mmsi)
                if mmsi in self.vessel_metadata.fishing_ranges_map:
                    for (start_time, end_time, is_fishing
                         ) in self.vessel_metadata.fishing_ranges_map[mmsi]:
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

    def build(self, net):
        self.logits = net
        self.prediction = tf.sigmoid(net)

    def build_trainer(self, timestamps, mmsis):
        update_ops = []

        dense_labels = self.dense_labels(
            tf.shape(self.prediction), timestamps, mmsis)
        thresholded_prediction = tf.to_int32(self.prediction > 0.5)
        valid = tf.to_int32(tf.not_equal(dense_labels, -1))
        ones = tf.to_int32(dense_labels > 0.5)
        weights = tf.to_float(valid)

        raw_loss = self.loss_function(dense_labels)

        update_ops.append(
            tf.summary.scalar('%s/Training-loss' % self.name, raw_loss))

        accuracy = slim.metrics.accuracy(
            thresholded_prediction, ones, weights=weights)
        update_ops.append(
            tf.summary.scalar('%s/Training-accuracy' % self.name, accuracy))

        loss = raw_loss * self.loss_weight

        return Trainer(loss, update_ops)

    def build_evaluation(self, timestamps, mmsis):
        dense_labels_fn = self.dense_labels

        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, prediction, timestamps,
                         mmsis, metrics):
                super(Evaluation, self).__init__(metadata_label, name,
                                                 prediction, timestamps, mmsis,
                                                 metrics)

            def build_test_metrics(self):
                dense_labels = dense_labels_fn(
                    tf.shape(self.prediction), self.timestamps, self.mmsis)
                thresholded_prediction = tf.to_int32(self.prediction > 0.5)
                valid = tf.to_int32(tf.not_equal(dense_labels, -1))
                ones = tf.to_int32(dense_labels > 0.5)
                weights = tf.to_float(valid)

                recall = slim.metrics.streaming_recall(
                    thresholded_prediction, ones, weights=weights)

                precision = slim.metrics.streaming_precision(
                    thresholded_prediction, ones, weights=weights)

                raw_metrics = {
                    'Test-MSE': slim.metrics.streaming_mean_squared_error(
                        self.prediction, tf.to_float(ones), weights=weights),
                    'Test-accuracy': slim.metrics.streaming_accuracy(
                        thresholded_prediction, ones, weights=weights),
                    'Test-precision': precision,
                    'Test-recall': recall,
                    'Test-F1-score': f1(recall, precision),
                    'Test-prediction-fraction':
                    slim.metrics.streaming_accuracy(
                        thresholded_prediction, valid, weights=weights),
                    'Test-label-fraction': slim.metrics.streaming_accuracy(
                        ones, valid, weights=weights)
                }

                return metrics.aggregate_metric_map(
                    {"{}/{}".format(self.name, k): v
                     for (k, v) in raw_metrics.items()})

            def build_json_results(self, prediction, timestamps):
                assert (len(prediction) == len(timestamps))
                thresholded_prediction = prediction > 0.5
                combined = zip(timestamps, thresholded_prediction)

                last = None
                fishing_ranges = []
                for ts_raw, is_fishing in combined:
                    ts = datetime.datetime.utcfromtimestamp(int(
                        ts_raw)).isoformat()
                    if last and last[0] >= ts:
                        break
                    if is_fishing:
                        if last and last[1]:
                            fishing_ranges[-1][1] = ts
                        else:
                            fishing_ranges.append([ts, ts])
                    last = (ts, is_fishing)

                return [{'start_time': start_time,
                         'end_time': end_time}
                        for (start_time, end_time) in fishing_ranges]

        return Evaluation(self.metadata_label, self.name, self.prediction,
                          timestamps, mmsis, self.metrics)


class FishingLocalizationObjectiveCrossEntropy(
        AbstractFishingLocalizationObjective):
    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 loss_weight=1.0,
                 metrics='all',
                 pos_weight=1.0):
        """

        args:
            metadata_label: label that we are classifying by.
            name: name of this objective (for metrics)
            vessel_metadata: info on classes for each mmsi loaded
            loss_weight: weight of this objective (so that can increase decrease relative to other objectives)
            metrics: which metrics to include. Options are currently ['all', 'minimal']
            pos_weight: increases the weight of getting positive values right, so an increased `pos_weight`
                        should increase recall at the expense of precision.

        """
        super(FishingLocalizationObjectiveCrossEntropy, self).__init__(
            metadata_label, name, vessel_metadata, loss_weight, metrics)
        self.pos_weight = pos_weight

    def loss_function(self, dense_labels):
        fishing_mask = tf.to_float(tf.not_equal(dense_labels, -1))
        fishing_targets = tf.to_float(dense_labels > 0.5)
        return (tf.reduce_mean(fishing_mask *
                               tf.nn.weighted_cross_entropy_with_logits(
                                   self.logits,
                                   fishing_targets,
                                   pos_weight=self.pos_weight)))
