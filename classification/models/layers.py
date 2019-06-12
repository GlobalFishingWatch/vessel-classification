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
import tensorflow.layers as ly
import numpy as np


def zero_pad_features(features, depth):
    """ Zero-pad features in the depth dimension to match requested feature depth. """

    n = int(features.get_shape().dims[-1])
    extra_feature_count = depth - n
    assert n >= 0
    if n > 0:
        padding = tf.tile(features[:, :, :1] * 0,
                          [1, 1, extra_feature_count])
        features = tf.concat([features, padding], 2)
    return features

def repeat_tensor(input, n):
    batch_size, width, depth = input.get_shape()
    repeated = tf.concat([input] * n, 2)
    return tf.reshape(repeated, [-1, int(width) * n, int(depth)])


def misconception_layer(inputs,
                        filters,
                        kernel_size,
                        strides,
                        training,
                        scope=None,
                        virtual_batch_size=None):
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
        extra = kernel_size - strides
        p0 = extra // 2
        p1 = extra - p0
        padded = tf.pad(inputs, [[0, 0], [p0, p1], [0, 0]])
        stage_conv = ly.conv1d(
            padded, filters, kernel_size, strides=strides, padding="valid", activation=None, use_bias=False)
        stage_conv = ly.batch_normalization(stage_conv, training=training, virtual_batch_size=virtual_batch_size)
        stage_conv = tf.nn.relu(stage_conv)
        stage_max_pool_reduce = tf.layers.max_pooling1d(
            padded, kernel_size, strides=strides, padding="valid")
        concat = tf.concat([stage_conv, stage_max_pool_reduce], 2)

        total = ly.conv1d(concat, filters, 1, activation=None, use_bias=False)
        total = ly.batch_normalization(total, training=training, virtual_batch_size=virtual_batch_size)
        total = tf.nn.relu(total)
        return total



def misconception_with_bypass(inputs,
                              filters,
                              kernel_size,
                              strides,
                              training,
                              scope=None,
                              virtual_batch_size=None):
    with tf.name_scope(scope):
        residual = misconception_layer(inputs, filters, kernel_size, strides, training, scope, virtual_batch_size)
        if strides > 1:
            inputs = tf.layers.max_pooling1d(
                inputs, strides, strides=strides, padding="valid")
        inputs = zero_pad_features(inputs, filters)
        return inputs + residual


def misconception_model(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5,
                        virtual_batch_size=None,
                        feature_means=None,
                        feature_stds=None):
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
    net = inputs
    if feature_means is not None:
        net = net - tf.constant(feature_means)[None, None, :]
    if feature_stds is not None:
        net = net / (tf.constant(feature_stds) + 1e-6)
    layers.append(net)
    for filters, strides in zip(filters_list, strides_list):
        net = misconception_with_bypass(net, filters, kernel_size, strides, training, virtual_batch_size=virtual_batch_size)
        layers.append(net)
    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training, virtual_batch_size=virtual_batch_size)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs, layers


def misconception_model_2(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5):
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
    net = inputs
    layers.append(net)
    for filters, strides in zip(filters_list, strides_list):
        net = misconception_with_bypass(net, filters, kernel_size, strides, training)
        layers.append(net)
    onet = net
    for _ in range(sub_layers - 1):
        onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
        onet = ly.batch_normalization(onet, training=training)
        onet = tf.nn.relu(onet)
    onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
    snet = ly.conv1d(onet, 1, 1, activation=tf.nn.relu)[:, :, 0]
    selector = tf.expand_dims(tf.nn.softmax(snet), 2)

    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)

        onet = onet * selector
        n = int(onet.get_shape().dims[1])
        onet = ly.average_pooling1d(onet, n, n)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs, layers


def misconception_fishing(inputs,
                          filters_list,
                          kernel_size,
                          strides_list,
                          objective_function,
                          training,
                          pre_filters=128,
                          post_filters=128,
                          post_layers=1,
                          dropout_rate=0.5,
                          internal_dropout_rate=0.5,
                          other_objectives=(),
                          feature_means=None,
                          feature_stds=None):

    _, layers = misconception_model(
        inputs,
        filters_list,
        kernel_size,
        strides_list,
        training,
        other_objectives,
        sub_filters=post_filters,
        sub_layers=2,
        dropout_rate=internal_dropout_rate,
        feature_means=feature_means,
        feature_stds=feature_stds
        )

    expanded_layers = []
    for i, lyr in enumerate(layers):
        lyr = ly.conv1d(lyr, pre_filters, 1, activation=None)
        lyr = ly.batch_normalization(lyr, training=training)
        lyr = tf.nn.relu(lyr)
        expanded_layers.append(repeat_tensor(lyr, 2**i))

    embedding = tf.add_n(expanded_layers)

    for _ in range(post_layers - 1):
        embedding = ly.conv1d(embedding, post_filters, 1, activation=None, use_bias=False)
        embedding = ly.batch_normalization(embedding, training=training)
        embedding = tf.nn.relu(embedding)

    embedding = ly.conv1d(embedding, post_filters, 1, activation=tf.nn.relu)
    embedding = ly.dropout(embedding, training=training, rate=dropout_rate)

    fishing_outputs = ly.conv1d(embedding, 1, 1, activation=None)

    return objective_function.build(fishing_outputs)



