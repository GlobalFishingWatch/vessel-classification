import abc
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics

TrainNetInfo = namedtuple("TrainNetInfo", ["optimizer", "objective_trainers"])


class ObjectiveBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name


class ObjectiveTrainer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.loss = None
        self.update_ops = []


class ClassificationObjective(ObjectiveBase):
    def __init__(self, name, metadata_label, classes, transformer=None):
        super(self.__class__, self).__init__(name)
        self.metadata_label = metadata_label
        self.classes = list(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.num_classes = len(classes)
        self.transformer = transformer

    def training_label(self, data_row):
        """ Return the index of this training label, or if it's unset, return
            -1 so the loss function can ignore the example.
        """
        label_value = data_row[self.metadata_label]
        if self.transformer:
            label_value = self.transformer(label_value)
        if label_value:
            return self.class_indices[label_value]
        else:
            return -1

    def build_trainer(self, logits, labels, loss_weight=1.0):
        class Trainer(ObjectiveTrainer):
            def __init__(self, name, num_classes, loss_weight, logits, labels):
                super(self.__class__, self).__init__()
                # Labels outside the one-hot num_classes range are just encoded
                # to all-zeros, so the use of -1 for unknown works fine here
                # when combined with a mask below.
                one_hot_labels = slim.one_hot_encoding(labels, num_classes)

                # Set the label weights to zero when we don't know the class.
                label_weights = tf.select(
                    tf.equal(labels, -1),
                    tf.zeros_like(
                        labels, dtype=tf.float32),
                    tf.ones_like(
                        labels, dtype=tf.float32))

                raw_loss = slim.losses.softmax_cross_entropy(
                    logits, one_hot_labels, weight=label_weights)
                self.loss = raw_loss * loss_weight
                class_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

                self.update_ops.append(
                    tf.scalar_summary('%s training loss' % name, raw_loss))

                accuracy = slim.metrics.accuracy(
                    labels, class_predictions, weights=label_weights)
                self.update_ops.append(
                    tf.scalar_summary('%s training accuracy' % name, accuracy))

        return Trainer(self.name, self.num_classes, loss_weight, logits,
                       labels)

    def build_evaluation(self, logits):
        class Evaluation(object):
            def __init__(self, name, classes, num_classes, logits):
                self.name = name
                self.classes = classes
                self.num_classes = num_classes
                self.softmax = slim.softmax(logits)

            def build_test_metrics(self, labels):
                predictions = tf.cast(tf.argmax(self.softmax, 1), tf.int32)

                label_mask = tf.select(
                    tf.equal(labels, -1), tf.zeros_like(labels),
                    tf.ones_like(labels))

                return metrics.aggregate_metric_map({
                    '%s test accuracy' % self.name: metrics.streaming_accuracy(
                        predictions, labels, weights=label_mask),
                })

        return Evaluation(self.name, self.classes, self.num_classes, logits)


class RegressionObjective(ObjectiveBase):
    pass


class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    batch_size = 32

    feature_duration_days = 180
    num_classes = 9
    max_sample_frequency_seconds = 5 * 60
    max_window_duration_seconds = feature_duration_days * 24 * 3600

    # We allocate a much smaller buffer than would fit the specified time
    # sampled at 5 mins intervals, on the basis that the sample is almost
    # always much more sparse.
    window_max_points = (max_window_duration_seconds /
                         max_sample_frequency_seconds) / 4

    min_viable_timeslice_length = 500

    def __init__(self, num_feature_dimensions):
        self.num_feature_dimensions = num_feature_dimensions
        self.training_objectives = None

    @abc.abstractmethod
    def build_training_net(self, features, labels):
        """Build net suitable for training model

        Args:
            features : queue
                features to feed into net
            labels : a dictionary of groundtruth labels for training
            fishing_timeseries_labels:
                groundtruth localisation of fishing

        Returns:
            TrainNetInfo

        """
        optimizer = trainers = None
        return loss, trainers

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
