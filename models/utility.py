import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading

def single_feature_file_reader(filename_queue, num_features):
  """ Read and interpret data from a TFRecord file. """
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

def np_array_random_fixed_time_extract(input_series, max_time_delta, output_length):
  input_shape = input_series.shape
  input_length = input_shape[0]
  max_offset = max(input_length - output_length, 0)
  if max_offset != 0:
    offset = np.random.randint(0, max_offset)
  else:
    offset = 0

  return np_array_fixed_time_extract(input_series[offset:], max_time_delta, output_length)

def np_array_fixed_time_extract(input_series, max_time_delta, output_length):
  """Extracts a fixed-time slice from a 2d numpy array, representing a time
  series.

  The input array must be 2d, representing a time series, with the first
  column representing a timestamp (sorted ascending). Any values in the series
  with a time greater than (first time + max_time_delta) are removed and the
  prefix series repeated into the window to pad.

  Args:
    input_series: the input series. A 2d array first column representing an ascending time.

    max_time_delta: the maximum value of a time point in the returned series.

    output_length: the number of points in the output series. Input series
        shorter than this will be repeated into the output series.

  Returns:
    An array of the same shape as the input, representing the fixed time slice.
  """
  max_time = input_series[0][0] + max_time_delta
  last_index = np.searchsorted(input_series[:, 0], max_time, side='right')
  input_series = input_series[:last_index]
  input_length = len(input_series)
  reps = int(np.ceil(output_length / float(input_length)))
  output_series = np.concatenate([input_series] * reps, axis=0)[:output_length]

  return output_series

def extract_features(input, max_time_delta, window_size):
  # Crop and pad to the specified time window.
  features = np_array_random_fixed_time_extract(input, max_time_delta, window_size)

  # Drop the first (timestamp) column
  features = features[:,1:]

  if not np.isfinite(features).all():
    logging.fatal('Bad features: %s', features)

  return features

def cropping_feature_file_reader(filename_queue, num_features, max_time_delta,
    window_size):
  context_features, sequence_features = single_feature_file_reader(filename_queue, num_features)

  movement_features = sequence_features['movement_features']
  label = tf.cast(context_features['vessel_type_index'], tf.int32)

  features = tf.py_func(lambda input: extract_features(input, max_time_delta, window_size),
      [movement_features], [tf.float32])

  return features, label


def inception_layer(input, window_size, stride, depth, scope=None):
  with tf.name_scope(scope):
    with slim.arg_scope([slim.conv2d],
                        padding = 'SAME',
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.3)):
      #stage_1_oned = slim.conv2d(input, depth, [1, 1])
      #stage_1_conv = slim.conv2d(stage_1_oned, depth, [1, window_size])
      #stage_1_conv_reduce = slim.conv2d(stage_1_conv, depth, [1, window_size], stride=[1, stride])

      stage_2_oned = slim.conv2d(input, depth, [1, 1])
      stage_2_conv = slim.conv2d(stage_2_oned, depth, [1, window_size])
      stage_2_max_pool_reduce = slim.max_pool2d(stage_2_conv, [1, window_size], stride=[1, stride],
              padding = 'SAME')

      #concat = tf.concat(3, [stage_1_conv_reduce, stage_2_max_pool_reduce])

      #return concat
      return stage_2_max_pool_reduce

def inception_model(input, window_size, stride, depth, levels, num_classes):
  with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
    net = slim.dropout(input, 0.5)
    for i in range(levels):
      net = inception_layer(net, window_size, stride, depth, "inception_%d" % i)
    #net = slim.repeat(net, levels, inception_layer, window_size, stride, depth)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 200)
    net = slim.dropout(net, 0.5)
    net = slim.fully_connected(net, 100)
    net = slim.dropout(net, 0.5)

    net = slim.fully_connected(net, num_classes) 

    return net

