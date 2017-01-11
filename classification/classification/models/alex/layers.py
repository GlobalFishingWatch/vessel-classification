import tensorflow as tf
import tensorflow.contrib.slim as slim


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
                activation_fn=tf.nn.elu,
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
    """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
    """
    with tf.name_scope(scope):
        misconception = misconception_layer(input, window_size, stride, depth,
                                            is_training, scope)
        bypass = slim.avg_pool2d(
            input, [1, window_size], stride=[1, stride], padding='SAME')

        return misconception + bypass


def misconception_model(input, window_size, stride, depth, levels,
                        objective_functions, is_training, dense_count=100, final_keep_prob=0.5):
    """ A misconception tower.

  Args:
    input: a tensor of size [batch_size, 1, width, depth].
    window_size: the width of the conv and pooling filters to apply.
    stride: the downsampling to apply when filtering.
    depth: the depth of the output tensor.
    levels: the height of the tower in misconception layers.
    objective_functions: a list of objective functions to add to the top of
                         the network.
    is_training: whether the network is training.

  Returns:
    a tensor of size [batch_size, num_classes].
  """
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
        net = input
        net = slim.repeat(net, levels, misconception_with_bypass, window_size,
                          stride, depth, is_training)
        net = slim.flatten(net)
        net = slim.dropout(net, final_keep_prob, is_training=is_training)
        net = slim.fully_connected(net, dense_count, normalizer_fn=slim.batch_norm, 
                                                     normalizer_params={'is_training': is_training})
        net = slim.dropout(net, final_keep_prob, is_training=is_training)

        outputs = [of.build(net) for of in objective_functions]

        return outputs
