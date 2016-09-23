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

def np_array_random_fixed_time_extract(rng, input_series, max_time_delta, output_length):
  input_length = len(input_series)
  start_time = input_series[0][0]
  end_time = input_series[-1][0]
  max_time_offset = max((end_time - start_time) - max_time_delta, 0)
  if max_time_offset == 0:
    time_offset = 0
  else:
    time_offset = rng(max_time_offset)
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
  def rng(upper):
    return np.random.randint(0, upper)
  features = np_array_random_fixed_time_extract(rng, input, max_time_delta, window_size)

  start_time = features[0][0]
  end_time = features[-1][0]

  # Drop the first (timestamp) column.
  features = features[:,1:]

  # Roll the features randomly to give different offsets.
  roll = np.random.randint(0, window_size)
  features = np.roll(features, roll, axis=0)

  if not np.isfinite(features).all():
    logging.fatal('Bad features: %s', features)

  return features, [start_time, end_time]

def cropping_feature_file_reader(filename_queue, num_features, max_time_delta,
    window_size):
  context_features, sequence_features = single_feature_file_reader(filename_queue, num_features)

  movement_features = sequence_features['movement_features']
  label = tf.cast(context_features['vessel_type_index'], tf.int32)
  weight = tf.cast(context_features['weight'], tf.float32)

  features, time_bounds = tf.py_func(lambda input: extract_features(input, max_time_delta, window_size),
      [movement_features], [tf.float32, tf.int64])

  return features, time_bounds, label


def extract_n_features(input, label, n, max_time_delta, window_size):
  samples = []
  for _ in range(n):
    features, time_bounds = extract_features(input, max_time_delta, window_size)
    samples.append((np.stack([features]), time_bounds, label))
  return zip(*samples)

def cropping_weight_replicating_feature_file_reader(filename_queue, num_features, max_time_delta,
    window_size, max_replication_factor):
  context_features, sequence_features = single_feature_file_reader(filename_queue, num_features)

  movement_features = sequence_features['movement_features']
  label = tf.cast(context_features['vessel_type_index'], tf.int32)
  weight = tf.cast(context_features['weight'], tf.float32)

  def replicate_extract(input, label, weight):
    n = int(np.ceil(float(max_replication_factor) * weight))
    return extract_n_features(input, label, n, max_time_delta, window_size)

  features_list, time_bounds_list, label_list = tf.py_func(replicate_extract,
      [movement_features, label, weight], [tf.float32, tf.float32, tf.int32])

  return features_list, time_bounds_list, label_list


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

