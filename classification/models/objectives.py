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
import tensorflow.metrics as metrics
from classification import metadata
import pytz
import six
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

    def create_metrics(self, labels):
        raw_metrics = self.create_raw_metrics(labels)
        try:
            eval_metrics = {"{}/{}".format(self.name, k) : v for (k, v) in raw_metrics.items()}
        except:
            logging.warning("Problem creating eval_metrics in {}".format(self))
            return {}
        for k, v in eval_metrics.items():
            tf.summary.scalar(k, v[1])
        return eval_metrics



class RegressionObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 value_from_id,
                 loss_weight=1.0,
                 metrics='all'):
        super(RegressionObjective, self).__init__(metadata_label, name,
                                                  loss_weight, metrics)
        self.value_from_id = value_from_id
        self.output_shape = []

    def create_label(self, id_, timestamps):
        self.value_from_id(id_)

    def build(self, net):
        self.prediction = tf.layers.dense(net, 1, activation=None)[:, 0]

    def expected_and_mask(self, labels):
        mask = ~tf.is_nan(labels)
        valid = tf.boolean_mask(labels, mask)
        idx = tf.to_int32(tf.where(mask))
        expected = tf.scatter_nd(idx, valid, tf.shape(labels))
        return expected, mask

    def masked_mean_error(self, labels):
        expected, mask = self.expected_and_mask(labels)
        mask = tf.cast(mask, tf.float32)
        count = tf.reduce_sum(mask)
        diff = tf.abs((expected - self.prediction) * mask)
        error = tf.reduce_sum(diff) / tf.maximum(count, EPSILON)
        return error

    def create_loss(self, labels):
        raw_loss = self._masked_mean_error(self.prediction, ids)
        return raw_loss * self.loss_weight

    def create_raw_metrics(self, labels):
        error = self.masked_mean_error(labels)
        loss = self.masked_mean_loss(self.prediction)
        return {
            'loss' : tf.metrics.mean(loss),
        }

   

class LogRegressionObjective(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 value_from_id,
                 loss_weight=1.0,
                 metrics='all'):
        super(LogRegressionObjective, self).__init__(metadata_label, name,
                                                     loss_weight, metrics)
        self.value_from_id = value_from_id
        self.output_shape = []

    def create_label(self, id_, timestamps):
        return self.value_from_id(id_)

    def build(self, net):
        self.prediction = tf.layers.dense(net, 1, activation=None)[:, 0]

    def expected_and_mask(self, labels):
        mask = ~tf.is_nan(labels)
        valid = tf.boolean_mask(labels, mask)
        idx = tf.to_int32(tf.where(mask))
        expected = tf.scatter_nd(idx, valid, tf.shape(labels))
        return expected, mask

    def masked_mean_loss(self, labels):
        expected, mask = self.expected_and_mask(labels)
        mask = tf.cast(mask, tf.float32)
        count = tf.reduce_sum(mask)
        squared_error = (
            (tf.log(expected + EPSILON) - self.prediction)**2 * mask)
        loss = tf.reduce_sum(squared_error) / tf.maximum(count, EPSILON)
        return loss

    def masked_mean_error(self, labels):
        expected, mask = self.expected_and_mask(labels)
        mask = tf.cast(mask, tf.float32)
        count = tf.reduce_sum(mask)
        diff = tf.abs((expected - tf.exp(self.prediction)) * mask)
        error = tf.reduce_sum(diff) / tf.maximum(count, EPSILON)
        return error

    def create_loss(self, labels):
        raw_loss = self.masked_mean_loss(labels)
        print(raw_loss, self.loss_weight)
        return raw_loss * self.loss_weight

    def create_raw_metrics(self, labels):
        loss = self.masked_mean_loss(labels)
        error = self.masked_mean_error(labels)
        return {
            'loss': tf.metrics.mean(loss),
            'error': tf.metrics.mean(error)
        }

    def build_json_results(self, prediction, timestamps):
        return {'name': self.name, 'value': np.exp(float(prediction))}



class LogRegressionObjectiveMAE(LogRegressionObjective):

    def __init__(self,
                 metadata_label,
                 name,
                 value_from_id,
                 loss_weight=1.0,
                 metrics='all'):
        super(LogRegressionObjectiveMAE, self).__init__(metadata_label, name, value_from_id,
                                                     loss_weight, metrics)

    def create_label(self, id_, timestamps):
        return self.value_from_id(id_)

    def masked_mean_loss(self, labels):
        expected, mask = self.expected_and_mask(labels)
        mask = tf.cast(mask, tf.float32)
        count = tf.reduce_sum(mask)
        mean_absolute_error = tf.abs(
            (tf.log(expected + EPSILON) - self.prediction) * mask)
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
        self.classes = metadata.VESSEL_CLASS_DETAILED_NAMES
        self.num_classes = metadata.multihot_lookup_table.shape[-1]
        self.class_indices = {k[0]: i for (i, k) in enumerate(metadata.VESSEL_CATEGORIES)}
        self.output_shape = [self.num_classes]


    def build(self, net):
        self.logits = tf.layers.dense(
            net, self.num_classes, activation=None)
        self.prediction = tf.nn.softmax(self.logits)

    def create_label(self, id_, timestamps):
        encoded = np.zeros([self.num_classes], dtype=np.int32)
        lbl_str = self.vessel_metadata.vessel_label('label', id_).strip()
        if lbl_str:
            for lbl in lbl_str.split('|'):
                j = self.class_indices[lbl]
                # Use '|' rather than '+' since classes might not be disjoint
                encoded |= metadata.multihot_lookup_table[j]
        return encoded.astype(np.float32)

    def create_loss(self, labels):
        with tf.variable_scope("custom-loss"):
            mask = tf.to_float(tf.greater_equal(tf.reduce_sum(labels, axis=1), 1))
            positives = tf.reduce_sum(
                tf.to_float(labels) * self.prediction, reduction_indices=[1])
            raw_loss = -tf.reduce_mean(mask * tf.log(positives + EPSILON))
        return raw_loss * self.loss_weight

    def create_raw_metrics(self, labels):
        mask = tf.to_float(tf.equal(tf.reduce_sum(labels, axis=1), 1))
        encoded_labels = tf.to_int32(tf.argmax(labels, axis=1))
        predictions = tf.to_int32(tf.argmax(self.prediction, axis=1))
        loss = self.create_loss(labels)
        return {
            'accuracy' : metrics.accuracy(predictions, encoded_labels, weights=mask),
            'loss' : tf.metrics.mean(loss)
                }

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


class FishingLocalizationObjectiveCrossEntropy(ObjectiveBase):
    def __init__(self,
                 metadata_label,
                 name,
                 vessel_metadata,
                 loss_weight=1.0,
                 metrics='all',
                 window=None):
        super(FishingLocalizationObjectiveCrossEntropy, self).__init__(metadata_label, name, loss_weight,
                               metrics)
        self.vessel_metadata = vessel_metadata
        self.window = window
        self.pos_weight = 1.0

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

    def build(self, net):
        self.logits = net[:, :, 0]
        self.prediction = tf.sigmoid(self.logits)

    def create_loss(self, dense_labels):
        return self.loss_weight * self.loss_function(dense_labels)

    def create_raw_metrics(self, dense_labels):
        thresholded_prediction = tf.to_int32(self.prediction > 0.5)
        valid = tf.to_int32(tf.not_equal(dense_labels, -1))
        labels = tf.to_int32(dense_labels > 0.5)
        weights = tf.to_float(valid)
        prediction = self.prediction

        if self.window:
            b, e = self.window
            prediction = prediction[:, b:e]
            dense_labels = dense_labels[:, b:e]
            thresholded_prediction = thresholded_prediction[:, b:e]
            valid = valid[:, b:e]
            labels = labels[:, b:e]
            weights = weights[:, b:e]

        return {
            'MSE': tf.metrics.mean_squared_error(prediction, dense_labels, weights=weights),
            'accuracy': tf.metrics.accuracy(labels, thresholded_prediction, weights=weights),
            'precision': tf.metrics.precision(labels, thresholded_prediction, weights=weights),
            'recall':    tf.metrics.recall(labels, thresholded_prediction, weights=weights)
        }



    def build_json_results(self, prediction, timestamps):
        InferencePoint = namedtuple('InferencePoint', ['timestamp', 'is_fishing'])
        InferenceRange = namedtuple('InferenceRange', ['start_time', 'end_time', 'score'])

        assert (len(prediction) == len(timestamps))
        thresholded_prediction = prediction > 0.5
        combined = list(six.moves.zip(timestamps, thresholded_prediction))
        if self.window:
            b, e = self.window
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









