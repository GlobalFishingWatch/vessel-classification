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
from .shake_shake import shake_shake, shake_out, shake_out2


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


def shake2(inputs,
          filters,
          kernel_size,
          stride,
          training,
          scope=None):

    with tf.name_scope(scope):

        y  = tf.nn.relu(inputs)

        y1 = ly.conv1d(y, filters, kernel_size, activation=None, use_bias=False, strides=stride, padding="valid")
        y1 = ly.batch_normalization(y1, training=training)
        y1 = tf.nn.relu(y1)
        y1 = ly.conv1d(y1, filters, 1, activation=None, use_bias=False, padding="valid")
        y1 = ly.batch_normalization(y1, training=training)

        y2 = ly.conv1d(y, filters, kernel_size, activation=None, use_bias=False, strides=stride, padding="valid")
        y2 = ly.batch_normalization(y2, training=training)
        y2 = tf.nn.relu(y2)
        y2 = ly.conv1d(y2, filters, 1, activation=None, use_bias=False, padding="valid")
        y2 = ly.batch_normalization(y2, training=training)

        return shake_shake(y1, y2, training)

def shakeout2(inputs,
          filters,
          kernel_size,
          stride,
          training,
          scope=None):

    with tf.name_scope(scope):

        y  = tf.nn.relu(inputs)

        y1 = ly.conv1d(y, filters, kernel_size, activation=None, use_bias=False, strides=stride, padding="valid")
        y1 = ly.batch_normalization(y1, training=training)
        y1 = tf.nn.relu(y1)
        y1 = ly.conv1d(y1, filters, 1, activation=None, use_bias=False, padding="valid")
        y1 = ly.batch_normalization(y1, training=training)

        y2 = ly.conv1d(y, filters, kernel_size, activation=None, use_bias=False, strides=stride, padding="valid")
        y2 = ly.batch_normalization(y2, training=training)
        y2 = tf.nn.relu(y2)
        y2 = ly.conv1d(y2, filters, 1, activation=None, use_bias=False, padding="valid")
        y2 = ly.batch_normalization(y2, training=training)

        return shake_out2(y1, y2, training)

def shakeout(inputs,
          filters,
          kernel_size,
          stride,
          training,
          scope=None):

    with tf.name_scope(scope):

        y  = tf.nn.relu(inputs)

        y1, y2 = shake_out(y, training)

        y1 = ly.conv1d(y1, filters, kernel_size, activation=None, use_bias=False, strides=stride, padding="valid")
        y1 = ly.batch_normalization(y1, training=training)
        y1 = tf.nn.relu(y1)
        y1 = ly.conv1d(y1, filters, 1, activation=None, use_bias=False, padding="valid")
        y1 = ly.batch_normalization(y1, training=training)

        y2 = ly.conv1d(y, filters, kernel_size, activation=None, use_bias=False, strides=stride, padding="valid")
        y2 = ly.batch_normalization(y2, training=training)
        y2 = tf.nn.relu(y2)
        y2 = ly.conv1d(y2, filters, 1, activation=None, use_bias=False, padding="valid")
        y2 = ly.batch_normalization(y2, training=training)

        return y1 + y2

def shake2_with_max(inputs,
                  filters,
                  kernel_size,
                  stride,
                  training,
                  scope=None):

    with tf.name_scope(scope):

        ss = shake2(inputs, filters, kernel_size, stride, training)

        mp = tf.layers.max_pooling1d(
            inputs, kernel_size, strides=stride, padding="valid")
        concat = tf.concat([ss, mp], 2)

        y = ly.conv1d(concat, filters, 1, activation=None, use_bias=False)
        y = ly.batch_normalization(y, training=training)

        return y


def shake2_with_bypass(inputs,
                              filters,
                              kernel_size,
                              stride,
                              training,
                              scope=None):

    with tf.name_scope(scope):

        residual = shake2(inputs, filters, kernel_size, stride, training)

        # crop = (kernel_size - stride // 2) // 2 # TODO: work this out more generally / cleanly
        crop = kernel_size // 2 
        thru = inputs
        if crop:
            thru = inputs[:, crop:-crop]
        thru = thru[:, ::stride]
        thru = zero_pad_features(thru, filters)

        return thru + residual


def shakeout2_with_bypass(inputs,
                              filters,
                              kernel_size,
                              stride,
                              training,
                              scope=None):

    with tf.name_scope(scope):

        residual = shakeout2(inputs, filters, kernel_size, stride, training)

        # crop = (kernel_size - stride // 2) // 2 # TODO: work this out more generally / cleanly
        crop = kernel_size // 2 
        thru = inputs
        if crop:
            thru = inputs[:, crop:-crop]
        thru = thru[:, ::stride]
        thru = zero_pad_features(thru, filters)

        return thru + residual


def shake2_with_thru_max(inputs,
                  filters,
                  kernel_size,
                  stride,
                  training,
                  scope=None):

    with tf.name_scope(scope):

        ss = shake2(inputs, filters, kernel_size, stride, training)

        mp = tf.layers.max_pooling1d(
            inputs, kernel_size, strides=stride, padding="valid")
        concat = tf.concat([ss, mp], 2)

        y = ly.conv1d(concat, filters, 1, activation=None, use_bias=False)
        residual = ly.batch_normalization(y, training=training)

        # crop = (kernel_size - stride // 2) // 2 # TODO: work this out more generally / cleanly
        crop = kernel_size // 2 
        thru = inputs
        if crop:
            thru = inputs[:, crop:-crop]
        thru = thru[:, ::stride]
        # if stride > 1:
        #     thru = tf.layers.max_pooling1d(
        #         thru, kernel_size, strides=stride, padding="valid")
        thru = zero_pad_features(thru, filters)

        return thru + residual


def shake2_with_bypass(inputs,
                              filters,
                              kernel_size,
                              stride,
                              training,
                              scope=None):

    with tf.name_scope(scope):

        residual = shake2(inputs, filters, kernel_size, stride, training)

        # crop = (kernel_size - stride // 2) // 2 # TODO: work this out more generally / cleanly
        crop = kernel_size // 2 
        thru = inputs
        if crop:
            thru = inputs[:, crop:-crop]
        thru = thru[:, ::stride]
        # if stride > 1:
        #     thru = tf.layers.max_pooling1d(
        #         thru, kernel_size, strides=stride, padding="valid")
        thru = zero_pad_features(thru, filters)

        return thru + residual


def shake2_model(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5,
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
    net = inputs

    if feature_means is not None:
        net = net - tf.constant(feature_means)[None, None, :]
    if feature_stds is not None:
        net = net / (tf.constant(feature_stds) + 1e-6)

    # Add a stem section to allow net to setup useful features
    filters = filters_list[0]
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    for filters, stride in zip(filters_list, strides_list):
        net = shake2_with_bypass(net, filters, kernel_size, stride, training)
    net = tf.nn.relu(net)

    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs


def shake2_max_model(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5,
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
    net = inputs

    if feature_means is not None:
        net = net - tf.constant(feature_means)[None, None, :]
    if feature_stds is not None:
        net = net / (tf.constant(feature_stds) + 1e-6)

    # Add a stem section to allow net to setup useful features
    filters = filters_list[0]
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    for filters, stride in zip(filters_list, strides_list):
        net = shake2_with_max(net, filters, kernel_size, stride, training)
    net = tf.nn.relu(net)

    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs


def shake2_thru_max_model(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5,
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
    net = inputs

    if feature_means is not None:
        net = net - tf.constant(feature_means)[None, None, :]
    if feature_stds is not None:
        net = net / (tf.constant(feature_stds) + 1e-6)

    # Add a stem section to allow net to setup useful features
    filters = filters_list[0]
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    for filters, stride in zip(filters_list, strides_list):
        net = shake2_with_thru_max(net, filters, kernel_size, stride, training)
    net = tf.nn.relu(net)

    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs


def shakeout_model(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5,
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
    net = inputs

    if feature_means is not None:
        net = net - tf.constant(feature_means)[None, None, :]
    if feature_stds is not None:
        net = net / (tf.constant(feature_stds) + 1e-6)

    # Add a stem section to allow net to setup useful features
    filters = filters_list[0]
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    for filters, stride in zip(filters_list, strides_list):
        net = shakeout2_with_bypass(net, filters, kernel_size, stride, training)
    net = tf.nn.relu(net)

    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs

def shake2_v2_model(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        final_layers,
                        final_filters,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5,
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
    net = inputs

    if feature_means is not None:
        net = net - tf.constant(feature_means)[None, None, :]
    if feature_stds is not None:
        net = net / (tf.constant(feature_stds) + 1e-6)

    # Add a stem section to allow net to setup useful features
    filters = filters_list[0]
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    for filters, stride in zip(filters_list, strides_list):
        net = shake2(net, filters, kernel_size, stride, training)

    for _ in range(final_layers):
        net = shake2_with_bypass(net, final_filters, kernel_size, 1, training)

    net = tf.nn.relu(net)

    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs


def shake2_v3_model(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        final_layers,
                        final_filters,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5,
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
    net = inputs

    if feature_means is not None:
        net = net - tf.constant(feature_means)[None, None, :]
    if feature_stds is not None:
        net = net / (tf.constant(feature_stds) + 1e-6)

    # Add a stem section to allow net to setup useful features
    filters = filters_list[0]
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    for filters, stride in zip(filters_list, strides_list):
        net = shake2_with_max(net, filters, kernel_size, stride, training)

    for _ in range(final_layers):
        net = shake2_with_bypass(net, final_filters, kernel_size, 1, training)

    net = tf.nn.relu(net)

    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs


def shake2_v4_model(inputs,
                        filters_list,
                        kernel_size,
                        strides_list,
                        training,
                        objective_functions,
                        sub_filters=128,
                        sub_layers=2,
                        dropout_rate=0.5,
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
    net = inputs

    if feature_means is not None:
        net = net - tf.constant(feature_means)[None, None, :]
    if feature_stds is not None:
        net = net / (tf.constant(feature_stds) + 1e-6)

    # Add a stem section to allow net to setup useful features
    filters = filters_list[0]
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = ly.conv1d(net, filters, 3, activation=None, use_bias=False, strides=1, padding="valid")
    net = ly.batch_normalization(net, training=training)
    net = tf.nn.relu(net)

    for filters, stride in zip(filters_list, strides_list):
        net = shake2_with_bypass(net, filters, kernel_size, stride, training)
    net = tf.nn.relu(net)

    outputs = []
    for ofunc in objective_functions:
        onet = net
        for _ in range(sub_layers - 1):
            onet = ly.conv1d(onet, sub_filters, 1, activation=None, use_bias=False)
            onet = ly.batch_normalization(onet, training=training)
            onet = tf.nn.relu(onet)
        onet = ly.conv1d(onet, sub_filters, 1, activation=tf.nn.relu)
        onet = ly.flatten(onet)
        #
        onet = ly.dropout(onet, training=training, rate=dropout_rate)
        outputs.append(ofunc.build(onet))

    return outputs
