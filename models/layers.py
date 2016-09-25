import tensorflow as tf
import tensorflow.contrib.slim as slim

def misconception_layer(input, window_size, stride, depth, is_training, scope=None):
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
    with slim.arg_scope([slim.conv2d],
                        padding = 'SAME',
                        activation_fn=tf.nn.elu):
      stage_conv = slim.conv2d(input, depth, [1, window_size], stride=[1, stride])
      stage_max_pool_reduce = slim.max_pool2d(input, [1, window_size], stride=[1, stride],
          padding = 'SAME')

      concat = tf.concat(3, [stage_conv, stage_max_pool_reduce])

      return slim.conv2d(concat, depth, [1, 1])

def misconception_with_bypass(input, window_size, stride, depth, is_training, scope=None):
  """ A misconception_layer added to its ave-pool down-sampled input (a la ResNet).
  """
  with tf.name_scope(scope):
    misconception = misconception_layer(input, window_size, stride, depth, is_training, scope)
    bypass = slim.avg_pool2d(input, [1, window_size], stride=[1, stride], padding='SAME')

    return misconception + bypass

def misconception_model(input, window_size, stride, depth, levels, num_classes, is_training):
  """ A misconception tower.

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
    net = slim.repeat(net, levels, misconception_with_bypass, window_size, stride, depth, is_training)
    net = slim.flatten(net)
    net = slim.dropout(net, 0.5, is_training=is_training)
    net = slim.fully_connected(net, 100)
    net = slim.dropout(net, 0.5, is_training=is_training)

    net = slim.fully_connected(net, num_classes)

    return net