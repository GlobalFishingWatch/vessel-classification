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
          'weight': tf.FixedLenFeature([], tf.float32),
          'vessel_type_index': tf.FixedLenFeature([], tf.int64),
          'vessel_type_name': tf.FixedLenFeature([], tf.string)
      },
      sequence_features = {
          'movement_features': tf.FixedLenSequenceFeature(shape=(num_features,),
              dtype=tf.float32)
      })

  return context_features, sequence_features

def np_array_random_fixed_time_extract_inferior(input_series, max_time_delta, output_length):
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

def np_array_random_fixed_time_extract(input_series, max_time_delta, output_length):
  input_length = len(input_series)
  start_time = input_series[0][0]
  end_time = input_series[-1][0]
  max_time_offset = max((end_time - start_time) - max_time_delta, 0)
  if max_time_offset == 0:
    time_offset = 0
  else:
    time_offset = np.random.randint(0, max_time_offset)
  start_index = np.searchsorted(input_series[:, 0], start_time + time_offset, side='left')

  # Cannot start closer than 500 points from the end
  start_index = min(start_index, max(0, input_length - 500))
  crop_end_time = min(input_series[start_index][0] + max_time_delta, end_time)

  end_index = np.searchsorted(input_series[:, 0], crop_end_time, side='right')
  
  cropped = input_series[start_index:end_index]
  cropped_length = len(cropped)
  logging.debug("%d, %d, %d, %d, %d, %d, %d, %d", input_length, start_time, end_time,
      max_time_offset, time_offset, start_index, end_index, cropped_length)
  reps = int(np.ceil(output_length / float(cropped_length)))
  output_series = np.concatenate([cropped] * reps, axis=0)[:output_length]

  return output_series

def extract_features(input, max_time_delta, window_size):
  # Crop and pad to the specified time window.
  features = np_array_random_fixed_time_extract(input, max_time_delta, window_size)

  # Drop the first (timestamp) column.
  features = features[:,1:]

  # Roll the features randomly to give different offsets.
  roll = np.random.randint(0, window_size)
  features = np.roll(features, roll, axis=0)

  if not np.isfinite(features).all():
    logging.fatal('Bad features: %s', features)

  return features

def cropping_feature_file_reader(filename_queue, num_features, max_time_delta,
    window_size):
  context_features, sequence_features = single_feature_file_reader(filename_queue, num_features)

  movement_features = sequence_features['movement_features']
  label = tf.cast(context_features['vessel_type_index'], tf.int32)
  weight = tf.cast(context_features['weight'], tf.float32)

  features = tf.py_func(lambda input: extract_features(input, max_time_delta, window_size),
      [movement_features], [tf.float32])

  return features, label

def misconception_layer(input, window_size, stride, depth, is_training, scope=None):
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
  with tf.name_scope(scope):
    misconception = misconception_layer(input, window_size, stride, depth, is_training, scope)
    bypass = slim.avg_pool2d(input, [1, window_size], stride=[1, stride], padding='SAME')

    #return slim.batch_norm(misconception + bypass, is_training=is_training)
    return misconception + bypass

def misconception_model(input, window_size, stride, depth, levels, num_classes, is_training):
  with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
    net = input
    net = slim.repeat(net, levels, misconception_with_bypass, window_size, stride, depth, is_training)
    net = slim.flatten(net)
    net = slim.dropout(net, 0.5, is_training=is_training)
    net = slim.fully_connected(net, 100)
    net = slim.dropout(net, 0.5, is_training=is_training)

    net = slim.fully_connected(net, num_classes) 

    return net

