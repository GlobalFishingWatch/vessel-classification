import abc
from collections import namedtuple

TrainNetInfo = namedtuple("TrainNetInfo", ["loss", "optimizer", "logits"])


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
            labels : queue
                groundtruth labels for training

        Returns:
            TrainNetInfo

        """
        loss = optimizer = logits = None
        return loss, optimizer, logits

    @abc.abstractmethod
    def build_inference_net(self, features):
        """Build net suitable for running inference on model

        Args:
            features : tensor
                queue of features to feed into net

        Returns:
            logits : tensor

        """
        logits = None
        return logits
