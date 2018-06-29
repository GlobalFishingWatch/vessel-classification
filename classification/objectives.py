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
import pytz
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

EPSILON = 1e-20


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
        diff = tf.abs((expected - predictions) * mask)

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


class LogRegressionObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 value_from_mmsi,
                 loss_weight=1.0,
                 metrics='all'):
        super(LogRegressionObjective, self).__init__(metadata_label, name,
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

    def _masked_mean_loss(self, predictions, mmsis):
        expected, mask = self._expected_and_mask(mmsis)
        count = tf.reduce_sum(mask)
        squared_error = (
            (tf.log(expected + EPSILON) - predictions)**2 * mask)

        loss = tf.reduce_sum(squared_error) / tf.maximum(count, EPSILON)

        return loss

    def _masked_mean_error(self, predictions, mmsis):
        expected, mask = self._expected_and_mask(mmsis)
        count = tf.reduce_sum(mask)
        diff = tf.abs((expected - tf.exp(predictions)) * mask)

        error = tf.reduce_sum(diff) / tf.maximum(count, EPSILON)

        return error

    def build_trainer(self, timestamps, mmsis):
        raw_loss = self._masked_mean_loss(self.prediction, mmsis)

        update_ops = []
        update_ops.append(
            tf.summary.scalar('%s/Training-loss' % self.name, raw_loss))

        loss = raw_loss * self.loss_weight

        return Trainer(loss, update_ops)

    def build_evaluation(self, timestamps, mmsis):
        class Evaluation(EvaluationBase):
            def __init__(self, metadata_label, name, masked_mean_loss,
                         masked_mean_error, prediction, metrics):
                super(Evaluation, self).__init__(metadata_label, name,
                                                 prediction, timestamps, mmsis,
                                                 metrics)
                self.masked_mean_loss = masked_mean_loss
                self.masked_mean_error = masked_mean_error
                self.mmsis = mmsis

            def build_test_metrics(self):
                loss = self.masked_mean_loss(self.prediction, self.mmsis)
                error = self.masked_mean_error(self.prediction, self.mmsis)

                return metrics.aggregate_metric_map({
                    '%s/Test-loss' % self.name: metrics.streaming_mean(loss),
                    '%s/Test-error' % self.name: metrics.streaming_mean(error)
                })

            def build_json_results(self, prediction, timestamps):
                return {'name': self.name, 'value': np.exp(float(prediction))}

        return Evaluation(self.metadata_label, self.name,
                          self._masked_mean_loss, self._masked_mean_error,
                          self.prediction, self.metrics)



class LogRegressionObjectiveMAE(LogRegressionObjective):

    def _masked_mean_loss(self, predictions, mmsis):
        expected, mask = self._expected_and_mask(mmsis)
        count = tf.reduce_sum(mask)
        mean_absolute_error = tf.abs(
            (tf.log(expected + EPSILON) - predictions) * mask)

        loss = tf.reduce_sum(mean_absolute_error) / tf.maximum(count, EPSILON)

        return loss



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
        self.logits = slim.fully_connected(
            net, self.num_classes, activation_fn=None)
        self.prediction = slim.softmax(self.logits)

    def build_from_logits(self, logits):
        self.logits = logits
        self.prediction = slim.softmax(self.logits)

    def training_label(self, mmsi, label):
        """ Return the index of this training label, or if it's unset, return
            -1 so the loss function can ignore the example.
        """
        return self.vessel_metadata.vessel_label(label, mmsi) or -1

    def multihot_labels(self, mmsis):

        class_count = len(utility.VESSEL_CLASS_DETAILED_NAMES)

        def labels_from_mmsis(seq, class_indices):
            encoded = np.zeros([len(seq), class_count], dtype=np.int32)
            for i, m in enumerate(seq):
                lbl_str = self.vessel_metadata.vessel_label('label', m).strip()
                if lbl_str:
                    for lbl in lbl_str.split('|'):
                        j = class_indices[lbl]
                        # Use '|' rather than '+' since classes might not be disjoint
                        encoded[i] |= utility.multihot_lookup_table[j]
            return encoded

        indices = {k[0]: i for (i, k) in enumerate(utility.VESSEL_CATEGORIES)}

        labels = tf.py_func(lambda x: labels_from_mmsis(x, indices), [mmsis],
                            [tf.int32])

        labels = tf.reshape(
            labels, shape=tf.concat([tf.shape(mmsis), [class_count]], 0))

        return labels

    def build_trainer(self, timestamps, mmsis):

        labels = self.multihot_labels(mmsis)

        with tf.variable_scope("custom-loss"):
            positives = tf.reduce_sum(
                tf.to_float(labels) * self.prediction, reduction_indices=[1])
            raw_loss = -tf.reduce_mean(tf.log(positives))

        mask = tf.to_float(tf.equal(tf.reduce_sum(labels, 1), 1))
        int_labels = tf.to_int32(tf.argmax(labels, 1))
        int_predictions = tf.to_int32(tf.argmax(self.prediction, 1))
        accuracy = metrics.accuracy(int_labels, int_predictions, weights=mask)

        loss = raw_loss * self.loss_weight


        update_ops = []
        update_ops.append(
            tf.summary.scalar('%s/Training-loss' % self.name, raw_loss))
        update_ops.append(
            tf.summary.scalar('%s/Training-accuracy' % self.name, accuracy))      

        return Trainer(loss, update_ops)

    def build_evaluation(self, timestamps, mmsis):

        logits = self.logits

        multihot_labels = self.multihot_labels

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

            def build_test_metrics(self):

                raw_labels = multihot_labels(self.mmsis)
                mask = tf.to_float(tf.equal(tf.reduce_sum(raw_labels, 1), 1))
                labels = tf.to_int32(tf.argmax(raw_labels, 1))

                predictions = tf.to_int32(tf.argmax(self.prediction, 1))

                metrics_map = {
                    '%s/Test-accuracy' % self.name: metrics.streaming_accuracy(
                        predictions, labels, weights=mask)
                }

                if self.metrics == 'all':
                    for i, cls in enumerate(self.classes):
                        cls_name = cls.replace(' ', '-')
                        trues = tf.to_int32(tf.equal(labels, i))
                        preds = tf.to_int32(tf.equal(predictions, i))
                        recall = metrics.streaming_recall(
                            preds, trues, weights=mask)
                        precision = metrics.streaming_precision(
                            preds, trues, weights=mask)
                        metrics_map["%s/Class-%s-Precision" %
                                    (self.name, cls_name)] = recall
                        metrics_map["%s/Class-%s-Recall" %
                                    (self.name, cls_name)] = precision
                        metrics_map["%s/Class-%s-F1-Score" % (
                            self.name, cls_name)] = f1(recall, precision)
                        metrics_map["%s/Class-%s-ROC-AUC" % (
                            self.name, cls_name)] = metrics.streaming_auc(
                                self.prediction[:, i], trues, weights=mask)

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


class MultiClassificationObjectiveSmoothed(MultiClassificationObjective):

    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 loss_weight=1.0,
                 metrics='all',
                 smoothing_coefficient=0.1):
        self.epsilon = smoothing_coefficient
        super(MultiClassificationObjectiveSmoothed, self).__init__(metadata_label, name, vessel_metadata, loss_weight,
                                                           metrics)

    def build_trainer(self, timestamps, mmsis):

        labels = self.multihot_labels(mmsis)

        with tf.variable_scope("custom-loss"):
            # Normal args are the totals for each correct value (as used in standard cross entropy)
            normal_args = tf.reduce_sum(tf.to_float(labels) * self.prediction, 
                                            reduction_indices=[1])

            # To encourage self consistency in the face of noise we also use a component where the args
            # is the largest prediction.
            consistent_args = tf.reduce_max(self.prediction, axis=[1])
            #
            positives = (1 - self.epsilon) * tf.log(normal_args) + self.epsilon * tf.log(consistent_args)

            raw_loss = -tf.reduce_mean(positives)

        mask = tf.to_float(tf.equal(tf.reduce_sum(labels, 1), 1))
        int_labels = tf.to_int32(tf.argmax(labels, 1))
        int_predictions = tf.to_int32(tf.argmax(self.prediction, 1))
        accuracy = metrics.accuracy(int_labels, int_predictions, weights=mask)

        loss = raw_loss * self.loss_weight


        update_ops = []
        update_ops.append(
            tf.summary.scalar('%s/Training-loss' % self.name, raw_loss))
        update_ops.append(
            tf.summary.scalar('%s/Training-accuracy' % self.name, accuracy))      

        return Trainer(loss, update_ops)



class AbstractFishingLocalizationObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 loss_weight=1.0,
                 metrics='all',
                 window=None, ):
        ObjectiveBase.__init__(self, metadata_label, name, loss_weight,
                               metrics)
        self.vessel_metadata = vessel_metadata
        self.window = window

    def dense_labels(self, template_shape, timestamps, mmsis):
        # Convert fishing range labels to per-point labels.
        def dense_fishing_labels(mmsis_array, timestamps_array):
            dense_labels_list = []
            for mmsi, timestamps in zip(mmsis_array, timestamps_array):
                dense_labels = np.zeros_like(timestamps, dtype=np.float32)
                dense_labels.fill(-1.0)
                if mmsi in self.vessel_metadata.fishing_ranges_map:
                    for (start_time, end_time, is_fishing
                         ) in self.vessel_metadata.fishing_ranges_map[mmsi]:
                        start_range = calendar.timegm(start_time.utctimetuple(
                        ))
                        end_range = calendar.timegm(end_time.utctimetuple())
                        mask = (timestamps >= start_range) & (
                            timestamps <= end_range)
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

        if self.window:
            b, e = self.window
            dense_labels = dense_labels[:, b:e]
            thresholded_prediction = thresholded_prediction[:, b:e]
            valid = valid[:, b:e]
            ones = ones[:, b:e]
            weights = weights[:, b:e]

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
        eval_window = self.window

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
                prediction = self.prediction
                unclear = tf.to_int32((self.prediction > 0.333) & (
                    self.prediction < 0.666))

                if eval_window:
                    b, e = eval_window
                    prediction = prediction[:, b:e]
                    dense_labels = dense_labels[:, b:e]
                    thresholded_prediction = thresholded_prediction[:, b:e]
                    valid = valid[:, b:e]
                    ones = ones[:, b:e]
                    weights = weights[:, b:e]
                    unclear = unclear[:, b:e]

                recall = slim.metrics.streaming_recall(
                    thresholded_prediction, ones, weights=weights)

                precision = slim.metrics.streaming_precision(
                    thresholded_prediction, ones, weights=weights)

                raw_metrics = {
                    'Test-MSE': slim.metrics.streaming_mean_squared_error(
                        prediction, tf.to_float(ones), weights=weights),
                    'Test-accuracy': slim.metrics.streaming_accuracy(
                        thresholded_prediction, ones, weights=weights),
                    'Test-precision': precision,
                    'Test-recall': recall,
                    'Test-F1-score': f1(recall, precision),
                    'Test-prediction-fraction':
                    slim.metrics.streaming_accuracy(
                        thresholded_prediction, valid, weights=weights),
                    'Test-unclear-fraction':
                    slim.metrics.streaming_accuracy(
                        unclear, valid, weights=weights),
                    'Test-label-fraction': slim.metrics.streaming_accuracy(
                        ones, valid, weights=weights)
                }

                return metrics.aggregate_metric_map(
                    {"{}/{}".format(self.name, k): v
                     for (k, v) in raw_metrics.items()})

            def build_json_results(self, prediction, timestamps):
                InferencePoint = namedtuple('InferencePoint', ['timestamp', 'is_fishing'])
                InferenceRange = namedtuple('InferenceRange', ['start_time', 'end_time', 'score'])

                assert (len(prediction) == len(timestamps))
                thresholded_prediction = prediction > 0.5
                combined = zip(timestamps, thresholded_prediction)
                if eval_window:
                    b, e = eval_window
                    combined = combined[b:e]

                last = None
                fishing_ranges = []
                for ts_raw, is_fishing in combined:
                    ts = datetime.datetime.utcfromtimestamp(int(ts_raw))
                    if last and last.timestamp >= ts:
                        logging.warning("last.timestamp >= timestamp")
                        break
                    if last and last.is_fishing == is_fishing:
                        if ts.date() > last.timestamp.date():
                            # We are crossing a day boundary here, so break into two ranges
                            end_of_day = datetime.datetime.combine(last.timestamp.date(), 
                                datetime.time(hour=23, minute=59, second=59))
                            # TODO: are we skipping a day here if gaps is multi day? Check
                            start_of_day = datetime.datetime.combine(ts.date(), 
                                datetime.time(hour=0, minute=0, second=0))
                            fishing_ranges[-1] = fishing_ranges[-1]._replace(
                                                    end_time=end_of_day.isoformat())
                            fishing_ranges.append(
                                InferenceRange(start_of_day.isoformat(), None, is_fishing))
                        fishing_ranges[-1] =  fishing_ranges[-1]._replace(end_time=ts.isoformat())
                    else:
                        # TODO, append min(half the distance to previous / next point)
                        # TODO, but maybe we should drop long ranges with no points
                        fishing_ranges.append(
                            InferenceRange(ts.isoformat(), ts.isoformat(), is_fishing))
                    last = InferencePoint(timestamp=ts, is_fishing=is_fishing)

                return [{'start_time': x.start_time + 'Z',
                         'end_time': x.end_time + 'Z', 'value': float(x.score)}
                        for x in fishing_ranges]

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
                 pos_weight=1.0,
                 window=None):
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
            metadata_label, name, vessel_metadata, loss_weight, metrics,
            window)
        self.pos_weight = pos_weight

    def loss_function(self, dense_labels):
        fishing_mask = tf.to_float(tf.not_equal(dense_labels, -1))
        fishing_targets = tf.to_float(dense_labels > 0.5)
        logits = self.logits
        if self.window:
            b, e = self.window
            fishing_mask = fishing_mask[:, b:e]
            fishing_targets = fishing_targets[:, b:e]
            logits = logits[:, b:e]
        return tf.reduce_sum(fishing_mask *
                             tf.nn.weighted_cross_entropy_with_logits(
                                 targets=fishing_targets,
                                 logits=logits,
                                 pos_weight=self.pos_weight))


class FishingLocalizationObjectiveFishingTime(
        AbstractFishingLocalizationObjective):
    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 loss_weight=1.0,
                 metrics='all',
                 window=None):
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
        super(FishingLocalizationObjectiveFishingTime, self).__init__(
            metadata_label, name, vessel_metadata, loss_weight, metrics,
            window)

    def build(self, net, dt):
        self.logits = net
        self.prediction = tf.sigmoid(net)
        self.dt = dt

    def loss_function(self, dense_labels):


        # When awake: want this to be dt * self.prediction, but do something with missing values?
        # Then compare etih dt * fishing_targets

        fishing_mask = tf.to_float(tf.not_equal(dense_labels, -1))
        fishing_targets = dense_labels
        # tf.to_float(dense_labels > 0.5)
        logits = self.logits
        dt = self.dt
        if self.window:
            b, e = self.window
            fishing_mask = fishing_mask[:, b:e]
            fishing_targets = fishing_targets[:, b:e]
            logits = logits[:, b:e]
            # dt[0] => preds[1]
            dt = dt[:, b+1:e+1]
            
        return tf.reduce_sum(dt * fishing_mask *
                             tf.nn.sigmoid_cross_entropy_with_logits(
                                 labels=fishing_targets,
                                 logits=logits))


class FishingLocalizationObjectiveSquaredError(
        AbstractFishingLocalizationObjective):
    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 loss_weight=1.0,
                 metrics='all',
                 pos_weight=1.0,
                 window=None):
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
        super(FishingLocalizationObjectiveSquaredError, self).__init__(
            metadata_label, name, vessel_metadata, loss_weight, metrics,
            window)
        self.pos_weight = pos_weight

    def loss_function(self, dense_labels):
        fishing_mask = tf.to_float(tf.not_equal(dense_labels, -1))
        fishing_targets = tf.to_float(dense_labels > 0.5)
        logits = self.logits
        if self.window:
            b, e = self.window
            fishing_mask = fishing_mask[:, b:e]
            fishing_targets = fishing_targets[:, b:e]
            logits = logits[:, b:e]
        return tf.reduce_sum(fishing_mask *
                             (fishing_targets - tf.sigmoid(logits))**2)

