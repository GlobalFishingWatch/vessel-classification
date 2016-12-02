from __future__ import print_function, division
import tensorflow as tf
import tensorflow.contrib.slim as slim


def weight_variable(shape, name='W', mean=0.0, stddev=0.1):
    initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
    return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name='b'):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)


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

def max_pool_layer(inputs, size, stride=None, padding="SAME", name="max-pool-layer"):
    if stride is None:
        stride = size
    return tf.nn.max_pool(
        inputs, [1, 1, size, 1],
        [1, 1, stride, 1],
        padding=padding,
        name=name)

def avg_pool_layer(inputs, size, stride=None, padding="SAME", name="avg-pool-layer"):
    if stride is None:
        stride = size
    return tf.nn.avg_pool(
        inputs, [1, 1, size, 1],
        [1, 1, stride, 1],
        padding=padding,
        name=name)

def l2_pool_layer(inputs, size, stride=None, padding="SAME", name="l2-pool-layer"):
    if stride is None:
        stride = size
    return tf.sqrt(tf.nn.avg_pool(
        tf.square(inputs), [1, 1, size, 1],
        [1, 1, stride, 1],
        padding=padding,
        name=name))


# def gated_pool_layer(inputs, size, count, stride=None, padding="SAME", name="gated-pool-layer"):
#     if stride is None:
#         stride = size
#     with tf.variable_scope(name):
#         pools = [max_pool_layer(inputs, size, stride, padding), 
#                  avg_pool_layer(inputs, size, stride, padding),
#                  l2_pool_layer(inputs, size, stride, padding)] 
#         selector = tf.nn.softmax(tf.nn.elu(conv1d_layer(inputs, size, count + 1, stride, padding)))
#         # # Harmonize the dimension
#         # The selector dimension is 4
#         pools = tf.pack(pools, axis=4)
#         # Broadcase features along axis 3
#         selector = tf.pack([selector], axis=3)
#         # Reduce across pools
#         return tf.reduce_sum(pools * selector, reduction_indices=3)


def gated_pool_layer(inputs, size, count, stride=None, padding="SAME", name="gated-pool-layer"):
    if stride is None:
        stride = size
    with tf.variable_scope(name):
        H, W, C = [int(x) for x in inputs.get_shape().dims[1:]]
        pool_weights = [
        tf.concat(2, [weight_variable([1, size, 1, 1], name="W{}".format(i))] * C) for i in range(count)]
        # weighted_pools = [tf.nn.depthwise_conv2d(inputs, W, strides=[1, 1, stride, 1], padding=padding) for W in pool_weights]
        weighted_pools = [tf.nn.depthwise_conv2d(inputs, 
            tf.nn.softmax(PW, dim=1), 
            strides=[1, stride, stride, 1], padding=padding) for PW in pool_weights]
        pools = [max_pool_layer(inputs, size, stride, padding), 
                 avg_pool_layer(inputs, size, stride, padding),
                 l2_pool_layer(inputs, size, stride, padding)] + weighted_pools
        selector = tf.nn.sigmoid(conv1d_layer(inputs, size, count + 3, stride, padding))
        # # Harmonize the dimension
        # The selector dimension is 4
        pools = tf.pack(pools, axis=4)
        # Broadcase features along axis 3
        selector = tf.pack([selector], axis=3)
        # Reduce across pools
        return tf.reduce_sum(pools * selector, reduction_indices=4)


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

        # XXX
        mean, var =  mean_var_train() #tf.cond(is_training, mean_var_train, mean_var_test)

        return tf.nn.batch_normalization(inputs, mean, var, beta, gamma,
                                         epsilon)

def feature_pool_layer(inputs, size):
        inputs = tf.transpose(inputs, [0, 1, 3, 2])
        inputs = tf.nn.max_pool(inputs, [1, 1, size, 1], [1, 1, size, 1],
        padding="SAME",
        data_format='NHWC')
        return tf.transpose(inputs, [0, 1, 3, 2])

def cfp_pool_layer(inputs, size, stride, count=None, padding="SAME", name="cfp-layer"):
    with tf.variable_scope(name):
        if count is None:
            count = int(inputs.get_shape().dims[-1])
        layer = conv1d_layer(inputs, size, size * count, stride, padding=padding)
        return feature_pool_layer(layer, size)

def conv_fp_layer(inputs, filter_count, is_training, filter_size=3, pool_size=3, padding="SAME", name="max-pool-layer"):
    with tf.variable_scope(name):
        # Input is a n_batch x width x 1 x n_filter
        #
        layer = conv1d_layer(inputs, filter_size, pool_size * filter_count, padding=padding, name="sample")
        layer = batch_norm(layer, is_training)
        layer = feature_pool_layer(layer, pool_size)
        layer = conv1d_layer(layer, 1, pool_size * filter_count, name="filter")
        layer = batch_norm(layer, is_training)
        layer = feature_pool_layer(layer, pool_size)
        return layer


def misconception_layer(inputs,
                        filter_count,
                        is_training,
                        filter_size=3,
                        decay=None,
                        padding="SAME",
                        name='misconception_layer'):
    decay_arg = {} if (decay is None) else {'decay': decay}
    with tf.variable_scope(name):
        # Input is a n_batch x width x 1 x n_filter
        #
        conv = conv1d_layer(inputs, filter_size, filter_count, padding=padding)
        #
        pool = tf.nn.max_pool(
            inputs, [1, 1, filter_size, 1], [1, 1, 1, 1],
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
