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

import tensorflow as tf
import tensorflow.contrib.slim as slim
from classification import utility
import numpy as np


def zero_pad_features(features, depth):
    """ Zero-pad features in the depth dimension to match requested feature depth. """

    n = int(features.get_shape().dims[-1])
    extra_feature_count = depth - n
    assert n >= 0
    if n > 0:
        padding = tf.tile(features[:, :, :, :1] * 0,
                          [1, 1, 1, extra_feature_count])
        features = tf.concat(3, [features, padding])
    return features


def misconception_layer(input,
                        window_size,
                        stride,
                        depth,
                        is_training,
                        scope=None):
    """ A single layer of the misconception convolutional network.

  Args:
    input: a tensor of size [batch_size, 1, width, depth]
    window_size: the width of the conv and pooling filters to apply.
    stride: the downsampling to apply when filtering.
    depth: the depth of the output tensor.

  Returns:
    a tensor of size [batch_size, 1, width/stride, depth].
  """
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
            stage_conv = slim.conv2d(
                input, depth, [1, window_size], stride=[1, stride])
            stage_max_pool_reduce = slim.max_pool2d(
                input, [1, window_size], stride=[1, stride], padding='SAME')

            concat = tf.concat(3, [stage_conv, stage_max_pool_reduce])

            return slim.conv2d(concat, depth, [1, 1])


def misconception_with_bypass(input,
                              window_size,
                              stride,
                              depth,
                              is_training,
                              scope=None):
    with tf.name_scope(scope):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}):
            residual = misconception_layer(input, window_size, stride, depth,
                                           is_training, scope)

            if stride > 1:
                input = slim.avg_pool2d(
                    input, [1, stride], stride=[1, stride], padding='SAME')

            input = zero_pad_features(input, depth)

            return input + residual


def misconception_model(input,
                        window_size,
                        depths,
                        strides,
                        objective_functions,
                        is_training,
                        dense_count=256,
                        dense_layers=2,
                        keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    layers = []
    with slim.arg_scope([slim.batch_norm], decay=0.999):
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            net = input
            layers.append(net)
            for depth, stride in zip(depths, strides):
                net = misconception_with_bypass(net, window_size, stride,
                                                depth, is_training)
                layers.append(net)
            # Global average pool
            n = int(net.get_shape().dims[1])
            net = slim.avg_pool2d(net, [1, n], stride=[1, n])
            net = slim.flatten(net)
            outputs = []
            for ofunc in objective_functions:
                onet = net
                for _ in range(dense_layers - 1):
                    onet = slim.fully_connected(
                        onet,
                        dense_count,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training})
                # Don't use batch norm on last layer, just use dropout.
                onet = slim.fully_connected(onet, dense_count)
                onet = slim.dropout(onet, keep_prob, is_training=is_training)
                outputs.append(ofunc.build(onet))

    return outputs, layers


def misconception_fishing(input,
                          window_size,
                          depths,
                          strides,
                          objective_function,
                          is_training,
                          dense_count=16,
                          dense_layers=2,
                          keep_prob=0.5,
                          internal_keep_prob=0.5,
                          other_objectives=()):

    _, layers = misconception_model(
        input,
        window_size,
        depths,
        strides,
        other_objectives,
        is_training,
        dense_count=dense_count,
        dense_layers=2,
        keep_prob=keep_prob)

    expanded_layers = []
    for i, lyr in enumerate(layers):
        expanded_layers.append(utility.repeat_tensor(lyr, 2**i))

    embedding = tf.concat(3, expanded_layers)
    embedding = slim.dropout(
        embedding, internal_keep_prob, is_training=is_training)

    for _ in range(dense_layers - 1):
        embedding = slim.conv2d(
            embedding,
            dense_count, [1, 1],
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training})
    embedding = slim.conv2d(
        embedding,
        dense_count, [1, 1],
        activation_fn=tf.nn.relu,
        normalizer_fn=None)
    embedding = slim.dropout(embedding, keep_prob, is_training=is_training)

    fishing_outputs = tf.squeeze(
        slim.conv2d(
            embedding, 1, [1, 1], activation_fn=None, normalizer_fn=None),
        squeeze_dims=[1, 3])

    return objective_function.build(fishing_outputs)
