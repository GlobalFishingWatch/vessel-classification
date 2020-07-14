import tensorflow as tf


def shake_shake(x1, x2, is_training):
    is_training = tf.constant(is_training, dtype=tf.bool)
    # create alpha and beta
    batch_size = tf.shape(x1)[0]
    # TODO: modifed for 1d, make more general or rename
    alpha = tf.random_uniform((batch_size, 1, 1))
    beta = tf.random_uniform((batch_size, 1, 1))
    # shake-shake during training phase
    def x_shake():
        return beta * x1 + (1 - beta) * x2 + tf.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)
    # even-even during testing phase
    def x_even():
        return 0.5 * x1 + 0.5 * x2
    return tf.cond(is_training, x_shake, x_even)


def shake_out(x, is_training):
    is_training = tf.constant(is_training, dtype=tf.bool)
    # create alpha and beta
    batch_size = tf.shape(x)[0]
    feature_depth = tf.shape(x)[2] # TODO: bulletproof
    alpha = tf.random_uniform((batch_size, 1, feature_depth))
    # shake-shake during training phase
    def x_shake():
        return alpha * x, (1 - alpha * x)
    # even-even during testing phase
    def x_even():
        return 0.5 * x, 0.5 * x
    return tf.cond(is_training, x_shake, x_even)


def shake_out2(x1, x2, is_training):
    is_training = tf.constant(is_training, dtype=tf.bool)
    # create alpha and beta
    batch_size = tf.shape(x1)[0]
    feature_depth = tf.shape(x1)[2] # TODO: bulletproof
    # TODO: modifed for 1d, make more general or rename
    alpha = tf.random_uniform((batch_size, 1, feature_depth))
    # shake-shake during training phase
    def x_shake():
        return alpha * x1 + (1 - alpha) * x2 
    # even-even during testing phase
    def x_even():
        return 0.5 * x1 + 0.5 * x2
    return tf.cond(is_training, x_shake, x_even)