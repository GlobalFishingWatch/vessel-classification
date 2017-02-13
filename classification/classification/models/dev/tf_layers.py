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

from __future__ import print_function, division
import tensorflow as tf
import tensorflow.contrib.slim as slim

def weight_variable(shape, name='W'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable('b', initializer=initial)


def leaky_rectify(x, alpha=0.01):
    with tf.name_scope('leaky-rectify'):
        return tf.maximum(alpha * x, x)


def dropout_layer(inputs, is_training, keep_prob=0.5, name='dropout-layer'):
    with tf.variable_scope(name):
        kp = tf.cond(is_training, lambda: tf.constant(keep_prob),
                     lambda: tf.constant(1.0))
        return tf.nn.dropout(inputs, kp)


def dense_layer(inputs, size, name='dense-layer'):
    with tf.variable_scope(name):
        assert len(inputs.get_shape().dims) == 2
        n = int(inputs.get_shape().dims[1])
        W = weight_variable([n, size])
        b = bias_variable([size])
        return tf.matmul(inputs, W) + b

# Batch, 1, Width, Depth


def conv1d_layer(inputs,
                 filter_size,
                 filter_count,
                 stride=1,
                 padding="SAME",
                 name='conv1d-layer'):
    with tf.variable_scope(name):
        h, w, n = [int(x) for x in inputs.get_shape().dims[1:]]
        assert h == 1
        W = weight_variable([1, filter_size, n, filter_count])
        b = bias_variable([filter_count])
        return (tf.nn.conv2d(
            inputs, W, strides=[1, 1, stride, 1], padding=padding) + b)


def separable_conv1d_layer(inputs,
                 filter_size,
                 filter_count,
                 filter_mult=1,
                 stride=1,
                 padding="SAME",
                 name='sep-conv1d-layer'):
    with tf.variable_scope(name):
        h, w, n = [int(x) for x in inputs.get_shape().dims[1:]]
        assert h == 1
        W_pw = weight_variable([1, filter_size, n, filter_mult], name='W_pw')
        W_dw = weight_variable([1, 1, filter_mult * n, filter_count], name='W_dw')
        b = bias_variable([filter_count])
        return (tf.nn.separable_conv2d(
            inputs, W_pw, W_dw, strides=[1, 1, stride, 1], padding=padding) + b)


def atrous_conv1d_layer(inputs,
                        filter_size,
                        filter_count,
                        rate=1,
                        padding="SAME",
                        name='conv1d-layer'):
    with tf.variable_scope(name):
        h, w, n = [int(x) for x in inputs.get_shape().dims[1:]]
        assert h == 1
        W = weight_variable([1, filter_size, n, filter_count])
        b = bias_variable([filter_count])
        return (tf.nn.atrous_conv2d(inputs, W, rate=rate, padding=padding) + b)


def batch_norm(inputs,
               is_training,
               alpha=0.5,
               test_decay=0.999,
               epsilon=1e-3,
               name='batch-norm'):
    """
    """
    n = int(inputs.get_shape().dims[-1])
    axes = [x for (x, _) in enumerate(inputs.get_shape().dims)][:-1]
    with tf.variable_scope(name):
        beta = tf.Variable(
            tf.constant(
                0.0, shape=[n]), name='beta', trainable=True)
        gamma = tf.Variable(
            tf.constant(
                1.0, shape=[n]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, axes, name='moments')
        test_ema = tf.train.ExponentialMovingAverage(decay=test_decay)

        def mean_var_train():
            test_apply_op = test_ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([test_apply_op]):
                return (tf.identity(alpha * batch_mean + (1 - alpha) *
                                    test_ema.average(batch_mean)),
                        tf.identity(alpha * batch_var + (1 - alpha) *
                                    test_ema.average(batch_var)))

        def mean_var_test():
            return test_ema.average(batch_mean), test_ema.average(batch_var)

        mean, var = tf.cond(is_training, mean_var_train, mean_var_test)

        return tf.nn.batch_normalization(inputs, mean, var, beta, gamma,
                                         epsilon)


def misconception_layer(inputs,
                        filter_count,
                        is_training,
                        filter_size=3,
                        stride=1,
                        decay=None,
                        padding="SAME",
                        name='misconception_layer'):
    decay_arg = {} if (decay is None) else {'decay': decay}
    with tf.variable_scope(name):
        # Input is a n_batch x width x 1 x n_filter
        #
        conv = tf.nn.elu(
                    batch_norm(
                        conv1d_layer(inputs, filter_size, filter_count, stride=stride, padding=padding),
                            is_training, **decay_arg))
        #
        pool = tf.nn.max_pool(
            inputs, [1, 1, filter_size, 1], [1, 1, stride, 1],
            padding=padding,
            data_format='NHWC')
        #
        joint = tf.concat(3, [conv, pool])
        #
        return conv1d_layer(
                    joint, 1, filter_count, name="NIN")




