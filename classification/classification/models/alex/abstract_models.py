from classification.model import ModelBase
from classification import utility
from . import layers

import tensorflow as tf
import tensorflow.contrib.slim as slim


class MisconceptionModel(ModelBase):
    def __init__(self, num_feature_dimensions, vessel_metadata):
        super(MisconceptionModel, self).__init__(num_feature_dimensions,
                                                 vessel_metadata)

    def zero_pad_features(self, features):
        """ Zero-pad features in the depth dimension to match requested feature depth. """

        feature_pad_size = self.feature_depth - self.num_feature_dimensions
        assert (feature_pad_size >= 0)
        zero_padding = tf.zeros(
            [self.batch_size, 1, self.window_max_points, feature_pad_size])
        padded = tf.concat(3, [features, zero_padding])

        return padded

class MisconceptionWithFishingRangesModel(MisconceptionModel):
    def __init__(self, num_feature_dimensions, vessel_metadata):
        super(MisconceptionWithFishingRangesModel, self).__init__(
            num_feature_dimensions, vessel_metadata)

    def misconception_with_fishing_ranges(self, input, mmsis, is_training):
        """ A misconception tower with additional fishing range classification.

        Args:
            input: a tensor of size [batch_size, 1, width, depth].
            window_size: the width of the conv and pooling filters to apply.
            stride: the downsampling to apply when filtering.
            depth: the depth of the output tensor.
            levels: The height of the tower in misconception layers.

        Returns:
            a tensor of size [batch_size, num_classes].
        """
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
            net = input

            # Then a tower for classification.
            multiscale_layers = []
            for i in range(self.levels):
                with tf.variable_scope("layer_%d" % i):
                    multiscale_layers.append(utility.repeat_tensor(net, 2**i))

                    net = layers.misconception_with_bypass(
                        net, self.window_size, self.stride, self.feature_depth,
                        is_training)

            net = slim.flatten(net)
            net = slim.dropout(net, 0.5, is_training=is_training)
            net = slim.fully_connected(net, 100)
            net = slim.dropout(net, 0.5, is_training=is_training)

            concatenated_multiscale_embedding = tf.concat(3, multiscale_layers)

            fishing_outputs = tf.squeeze(
                slim.conv2d(
                    concatenated_multiscale_embedding,
                    1, [1, 1],
                    activation_fn=None),
                squeeze_dims=[1, 3])

            for of in self.classification_training_objectives:
                of.build(net)

            self.fishing_localisation_objective.build(fishing_outputs)
