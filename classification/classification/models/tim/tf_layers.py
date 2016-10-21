from __future__ import print_function, division
import tensorflow as tf
import tensorflow.contrib.slim as slim


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable("W", initializer=initial)


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
                        filter_rate=1,
                        decay=None,
                        padding="SAME",
                        name='misconception_layer'):
    decay_arg = {} if (decay is None) else {'decay': decay}
    with tf.variable_scope(name):
        # Input is a n_batch x width x 1 x n_filter
        #
        conv = atrous_conv1d_layer(inputs, filter_size, filter_count, filter_rate, padding=padding)
        #
        pool_width = filter_rate * (filter_size - 1) + 1
        pool = tf.nn.max_pool(
            inputs, [1, 1, pool_width, 1], [1, 1, 1, 1],
            padding=padding,
            data_format='NHWC')
        #
        joint = tf.nn.elu(
            batch_norm(tf.concat(3, [conv, pool]), is_training, **decay_arg))
        #
        return tf.nn.elu(
            batch_norm(
                conv1d_layer(
                    joint, 1, filter_count, name="NIN"),
                is_training,
                **decay_arg))
