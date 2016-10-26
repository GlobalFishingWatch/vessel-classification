import abc
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics

TrainNetInfo = namedtuple("TrainNetInfo", [
    "loss", "optimizer", "vessel_class_logits", "fishing_localisation_logits"
])


class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    batch_size = 32

    feature_duration_days = 180
    num_classes = 9
    num_feature_dimensions = 9
    max_sample_frequency_seconds = 5 * 60
    max_window_duration_seconds = feature_duration_days * 24 * 3600

    # We allocate a much smaller buffer than would fit the specified time
    # sampled at 5 mins intervals, on the basis that the sample is almost
    # always much more sparse.
    window_max_points = (max_window_duration_seconds /
                         max_sample_frequency_seconds) / 4

    min_viable_timeslice_length = 500

    @abc.abstractmethod
    def build_training_net(self, features, labels, fishing_timeseries_labels):
        """Build net suitable for training model

        Args:
            features : queue
                features to feed into net
            labels : queue
                groundtruth labels for training
            fishing_timeseries_labels:
                groundtruth localisation of fishing

        Returns:
            TrainNetInfo

        """
        loss = optimizer = vessel_class_logits = fishing_localisation_logits = None
        return loss, optimizer, vessel_class_logits, fishing_localisation_logits

    @abc.abstractmethod
    def build_inference_net(self, features):
        """Build net suitable for running inference on model

        Args:
            features : tensor
                queue of features to feed into net

        Returns:
            vessel_class_logits : tensor with vessel classes for the batch
            fishing_localisation_logits: tensor with fishing localisation scores
                                         for the batch

        """
        vessel_class_logits = fishing_localisation_logits = None
        return vessel_class_logits, fishing_localisation_logits


    def add_training_summaries(self, loss, logits, labels, fishing_localisation_logits, fishing_timeseries_labels):

        tf.scalar_summary('Training loss', loss)

        vessel_class_predictions = tf.cast(
            tf.argmax(logits, 1), tf.int32)

        vessel_class_accuracy = slim.metrics.accuracy(labels,
                                                      vessel_class_predictions)
        tf.scalar_summary('Vessel class training accuracy',
                          vessel_class_accuracy)



    def create_evaluate_metric_map(self, logits, labels, fishing_localisation_logits, fishing_timeseries_labels):

        vessel_class_predictions = tf.cast(
            tf.argmax(logits, 1), tf.int32)

        return {
            'Vessel class test accuracy':
            metrics.streaming_accuracy(vessel_class_predictions, labels),
            }



class MixedFishingModelBase(ModelBase):


    def add_training_summaries(self, loss, logits, labels, fishing_localisation_logits, fishing_timeseries_labels):


        ModelBase.add_training_summaries(self, loss, logits, labels, fishing_localisation_logits, fishing_timeseries_labels)

        fishing_timeseries_scores = tf.nn.sigmoid(fishing_localisation_logits)
        fishing_mask = tf.to_float(tf.not_equal(fishing_timeseries_labels, -1))

        fishing_delta = fishing_mask * (fishing_timeseries_scores - fishing_timeseries_labels)
        unscaled_fishing_loss = tf.reduce_sum(fishing_delta ** 2)
        fishing_loss_scale = tf.reduce_sum(tf.to_float(tf.not_equal(fishing_timeseries_labels, -1))) + 1e-6
        fishing_loss = unscaled_fishing_loss / fishing_loss_scale

        tf.scalar_summary('Fishing Score Count',
                          fishing_loss_scale)

        tf.scalar_summary('Fishing Score MSE',
                          fishing_loss)


    def create_evaluate_metric_map(self, logits, labels, fishing_localisation_logits, fishing_timeseries_labels):

        metric_map = ModelBase.create_evaluate_metric_map(self, logits, labels, fishing_localisation_logits, fishing_timeseries_labels)

        fishing_timeseries_scores = tf.nn.sigmoid(fishing_localisation_logits)

        fishing_timeseries_predictions = tf.to_int32(fishing_timeseries_scores > 0.5)
        fishing_timeseries_ones = tf.to_int32(fishing_timeseries_labels > 0.5)
        fishing_weights = tf.to_float(tf.not_equal(fishing_timeseries_labels, -1))

        metric_map.update({
            'Fishing score test MSE':
            slim.metrics.streaming_mean_squared_error(fishing_timeseries_predictions, fishing_timeseries_ones, 
                            weights=fishing_weights),
            'Fishing score test accuracy':
            slim.metrics.streaming_accuracy(fishing_timeseries_predictions, fishing_timeseries_ones, 
                            weights=fishing_weights),
            'Fishing score test precision':
            slim.metrics.streaming_precision(fishing_timeseries_predictions, fishing_timeseries_ones, 
                            weights=fishing_weights),
            'Fishing score test recall':
            slim.metrics.streaming_recall(fishing_timeseries_predictions, fishing_timeseries_ones, 
                            weights=fishing_weights),
            'Fishing score test fishing fraction':
            slim.metrics.streaming_accuracy(fishing_timeseries_ones, tf.to_int32(fishing_weights), 
                            weights=fishing_weights)
        })


        return metric_map

