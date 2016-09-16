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
  input_shape = input_series.shape
  input_length = input_shape[0]
  output_series = np.zeros((output_length, input_shape[1]), dtype=np.float32)
  input_index = 0
  max_time = input_series[0][0] + max_time_delta
  for output_index in range(output_length):
    ins = input_series[input_index]
    if ins[0] > max_time:
      input_index = 0
      ins = input_series[input_index]
    output_series[output_index] = ins
    if input_index < (input_length-1):
      input_index += 1
    else:
      input_index = 0

  return output_series

def cropping_feature_file_reader(filename_queue, num_features, max_time_delta,
    window_size):
  context_features, sequence_features = single_feature_file_reader(filename_queue, num_features)

  movement_features = sequence_features['movement_features']
  label = tf.cast(context_features['vessel_type_index'], tf.int32)

  def crop(input):
    return np_array_fixed_time_extract(input, max_time_delta, window_size)

  cropped = tf.py_func(crop, [movement_features], [tf.float32])

  return cropped, label

class CroppingFeatureReader(object):
  def __init__(self, filename_queue, num_feature_dimensions):
    self.reader = single_feature_file_reader(filename_queue,
        num_feature_dimensions)
    self.num_feature_dimensions = num_feature_dimensions

  def read(self, session, output_feature_queue, max_time_delta, window_length):
    context_features, sequence_features = session.run(self.reader)
    movement_features = sequence_features['movement_features']
    mmsi = tf.cast(context_features['mmsi'], tf.int32)
    label = tf.cast(context_features['vessel_type_index'], tf.int32)
    label_name = tf.cast(context_features['vessel_type_name'], tf.string)

    logging.info("Reading %d with %s", session.run(mmsi),
        session.run(label_name))

    cropped = tf.convert_to_tensor(np_array_fixed_time_extract(movement_features, max_time_delta,
        window_length), dtype=tf.float32)

    session.run(output_feature_queue.enqueue(([cropped], label)))

  def run(self, session, output_feature_queue, max_time_delta, window_length):
    def _run_loop():
      logging.info("Runner thread started.")
      while True:
        logging.info("Read.")
        self.read(session, output_feature_queue, max_time_delta, window_length)
    threading.Thread(target=_run_loop).start()

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



def inception_layer(input, window_size, stride, depth, scope=None):
  with tf.name_scope(scope):
    with slim.arg_scope([slim.conv2d],
                        padding = 'SAME',
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
        stage_1_oned = slim.conv2d(input, depth, [1, 1])
        stage_1_conv = slim.conv2d(stage_1_oned, depth, [1, window_size])
        stage_1_conv_reduce = slim.conv2d(stage_1_conv, depth, [1, window_size], stride=[1, stride])

        stage_2_oned = slim.conv2d(input, depth, [1, 1], stride=[1, 1])
        stage_2_conv = slim.conv2d(stage_2_oned, depth, [1, window_size])
        stage_2_max_pool_reduce = slim.max_pool2d(stage_2_conv, [1, window_size], stride=[1, stride],
                padding = 'SAME')

        concat = tf.concat(3, [stage_1_conv_reduce, stage_2_max_pool_reduce])

        return concat

def inception_model(input, window_size, stride, depth, levels, num_classes):
  with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
    net = slim.dropout(input, 0.5)
    net = slim.repeat(net, levels, inception_layer, window_size, stride, depth)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 200)
    net = slim.dropout(net, 0.5)
    net = slim.fully_connected(net, 100)
    net = slim.dropout(net, 0.5)

    net = slim.fully_connected(net, num_classes, activation_fn=None) 

    return net

