import abc
from collections import namedtuple

TrainNetInfo = namedtuple("TrainNetInfo", ["loss", "optimizer", "logits"])


class ModelBase(object):
    __metaclass__ = abc.ABCMeta

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
