import tensorflow as tf
import tensorflow.contrib.slim as slim

def feature_file_reader(input_file_pattern, num_features):
  input_files = tf.matching_files(input_file_pattern)
  filename_queue = tf.train.string_input_producer(input_files, num_epochs=None)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  # The serialized example is converted back to actual values.
  context_features, sequence_features = tf.parse_single_sequence_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      context_features = {
          'mmsi': tf.FixedLenFeature([], tf.int64),
          'vessel_type_index': tf.FixedLenFeature([], tf.int64),
          'vessel_type_name': tf.FixedLenFeature([], tf.string)
      },
      sequence_features = {
          'movement_features': tf.FixedLenSequenceFeature(shape=(num_features,),
              dtype=tf.float32)
      })

  return context_features, sequence_features


def fixed_time_extract(input_series, max_time_delta, output_length):
  """Extracts a fixed-time slice from a tensor.

  The input tensor must be 2d, representing a time series, with the first
  column representing a timestamp (sorted ascending). Any values in the series
  with a time greater than (first time + max_time_delta) are removed and the
  prefix series repeated into the window to pad.

  Args:
    input_series: the input series. We assume this is a 2d tensor, with the
        first column representing an ascending time.

    max_time_delta: the maximum value of a time point in the returned series.

    output_length: the number of points in the output series. Input series
        shorter than this will be repeated into the output series.

  Returns:
    A tensor of the same shape as the input, representing the fixed time slice.
  """
  length = tf.gather(tf.shape(input_series), 0)
  start = tf.slice(input_series, [0, 0], [0, -1])
  
  start_time = tf.gather(tf.gather(input_series, 0), 0)
  max_time = start_time + max_time_delta

  def update(i, index, t):
    element = tf.gather(input_series, index)
    element_timestamp = tf.gather(element, 0)

    index_inc = tf.add(index, 1)
    
    next_index = tf.cond(tf.logical_and(element_timestamp < max_time, index_inc < length),
        lambda: index_inc, lambda: tf.constant(0))
    return [tf.add(i, 1), next_index, tf.concat(0, [t, tf.pack([element])])]

  def condition(i, index, t):
    return tf.less(i, output_length)

  _, _, result = tf.while_loop(condition, update, [tf.constant(0), tf.constant(0), start])

  return result

def random_fixed_time_extract(input_series, max_time_delta, output_length):
  dimensions = tf.gather(tf.shape(input_series), 1)
  length = tf.gather(tf.shape(input_series), 0)
  max_crop_length = tf.minimum(output_length, length)
  cropped = tf.random_crop(input_series, [max_crop_length, dimensions])

  return fixed_time_extract(cropped, max_time_delta, output_length)


def inception_layer(input, window_size, stride, depth):
  stage_1_oned = slim.conv2d(input, depth, [1, 1])
  stage_1_conv = slim.conv2d(stage_1_oned, depth, [1, window_size])
  stage_1_conv_reduce = slim.conv2d(stage_1_conv, depth, [1, window_size], stride=[1, stride])

  stage_2_oned = slim.conv2d(input, depth, [1, 1], stride=[1, 1])
  stage_2_conv = slim.conv2d(stage_2_oned, depth, [1, window_size])
  stage_2_conv_reduce = slim.conv2d(stage_2_conv, depth, [1, window_size], stride=[1, stride])

  concat = tf.concat(3, [stage_1_conv_reduce, stage_2_conv_reduce])

  return concat

def inception_model(input, window_size, stride, depth, levels):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.tanh,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
    net = slim.dropout(input, 0.05)
    net = slim.repeat(net, levels, inception_layer, window_size, stride, depth)
    net = slim.fully_connected(net, 200, activation_fn=tf.nn.relu)
    net = slim.dropout(net, 0.5)
    net = slim.fully_connected(net, 100, activation_fn=tf.nn.relu)
    net = slim.dropout(net, 0.5)

    return net

