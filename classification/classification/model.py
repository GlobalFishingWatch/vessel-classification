import abc
from collections import namedtuple

TrainNetInfo = namedtuple("TrainNetInfo", ["optimizer", "objective_trainers"])


class ObjectiveBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, loss_weight):
        self.name = name
        self.loss_weight = loss_weight

    class Trainer(object):
        __metaclass__ = abc.ABCMeta

        def __init__(self):
            self.loss = None
            self.update_ops = []


class ClassificationObjective(ObjectiveBase):
    def __init__(self, name, num_classes, loss_weight=1.0):
        super(ClassificationObjective).__init__(self, name, loss_weight)
        self.num_classes = num_classes

    def build_trainer(self, logits, labels):
        class Trainer(ObjectiveTrainer):
            def __init__(self, name, num_classes, loss_weight, logits, labels):
                one_hot_labels = slim.one_hot_encoding(
                    labels, self.num_classes, weight=self.loss_weight)
                self.loss = slim.losses.softmax_cross_entropy(logits,
                                                              one_hot_labels)
                class_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

                self.update_ops.append(
                    tf.scalar_summary('%s training loss' % name, self.loss))

                accuracy = slim.metrics.accuracy(labels, class_predictions)
                self.update_ops.append(
                    tf.scalar_summary('%s training accuracy' % self.name, accuracy))

        return Trainer(self.name, self.num_classes, self.loss_weight, logits,
                       labels)


class RegressionObjective(ObjectiveBase):
    def __init__(self, name):
        super(RegressionObjective).__init__(self, name)

    class Trainer(ObjectiveTrainer):
        def __init__(self, name, predictions, targets):
            self.loss = slim.losses.mean_squared_error(
                predictions, targets, weight=self.loss_weight)

            tf.scalar_summary('%s training MSE' % self.name, self.loss)


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
        loss = optimizer = None
        return loss, optimizer

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
