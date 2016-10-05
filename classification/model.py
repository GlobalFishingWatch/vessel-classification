import abc


class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def build_training_net(self, features, labels):
        """Build net suitable for training model

		Args:
			features : tensor
				queue of features to feed into net
			labels : tensor
				queue of groundtruth labels for training

		Returns:
			(loss, optimizer, logits):

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
        loss = None
        return loss
