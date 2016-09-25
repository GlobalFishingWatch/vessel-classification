import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading

class ClusterNodeConfig(object):
  """ Class that represent the configuration of this node in a cluster. """

  def __init__(self, config):
    task_spec = config['task']
    self.task_type = task_spec['type']
    self.task_index = task_spec['index']
    self.cluster_spec = config['cluster']

  def is_master(self):
    return self.task_type == 'master'

  def is_worker(self):
    return self.task_type == 'worker'

  def is_ps(self):
    return self.task_type == 'ps'

  def is_chief(self):
    return task_index == 0

  def create_server(self):
    server = tf.train.Server(self.cluster_spec,
        job_name=self.task_type, task_index=self.task_index)

    logging.info("Server target: %s", server.target)
    return server

  @staticmethod
  def create_local_server_config():
    return ClusterNodeConfig({"cluster": {}, "task" : {"type": "worker", "index": 0}})


def single_feature_file_reader(filename_queue, num_features):
  """ Read and interpret data from a set of TFRecord files.

  Args:
    filename_queue: a queue of filenames to read through.
    num_features: the depth of the features.

  Returns:
    A pair of tuples:
      1. a context dictionary for the feature
      2. the vessel movement features, tensor of dimension [width, num_features].
  """

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

def np_array_random_fixed_time_extract(rng, input_series, max_time_delta,
    output_length, min_timeslice_size):
  """ Extracts a random fixed-time slice from a 2d numpy array.
    
  The input array must be 2d, representing a time series, with the first    
  column representing a timestamp (sorted ascending). Any values in the series    
  with a time greater than (first time + max_time_delta) are removed and the    
  prefix series repeated into the window to pad.    
    
  Args:
    rng: a single-arg function taking an int and uniformly returning a random
      number in the range [0, arg).
    input_series: the input series. A 2d array first column representing an
      ascending time.   
    max_time_delta: the maximum duration of the returned timeseries in seconds.
    output_length: the number of points in the output series. Input series    
      shorter than this will be repeated into the output series.   
    min_timeslice_size: the minimum number of points in a timeslice for the
      series to be considered meaningful. 
    
  Returns:    
    An array of the same depth as the input, but altered width, representing
    the fixed time slice.   
  """

  input_length = len(input_series)
  start_time = input_series[0][0]
  end_time = input_series[-1][0]
  max_time_offset = max((end_time - start_time) - max_time_delta, 0)
  if max_time_offset == 0:
    time_offset = 0
  else:
    time_offset = rng(max_time_offset)
  start_index = np.searchsorted(input_series[:, 0], start_time + time_offset, side='left')

  # Should not start closer than min_timeslice_size points from the end lest the 
  # series have too few points to be meaningful.
  start_index = min(start_index, max(0, input_length - min_timeslice_size))
  crop_end_time = min(input_series[start_index][0] + max_time_delta, end_time)

  end_index = np.searchsorted(input_series[:, 0], crop_end_time, side='right')
  
  cropped = input_series[start_index:end_index]
  cropped_length = len(cropped)
  reps = int(np.ceil(output_length / float(cropped_length)))
  output_series = np.concatenate([cropped] * reps, axis=0)[:output_length]

  return output_series

def np_array_extract_features(input, max_time_delta, window_size,
    min_timeslice_size):
  """ Extract and process a random timeslice from vessel movement features.

  Removes the timestamp column from the features, and applies a random roll to
  the chosen timeslice to further augment the training data.

  Args:
    input: the input data as a 2d numpy array.

  Returns:
    A tuple comprising:
      1. The extracted feature timeslice.
      2. The start and end time of the timeslice (in int64 seconds since epoch).
  """
  features = np_array_random_fixed_time_extract(
      lambda upper: np.random.randint(0, upper), input,
      max_time_delta, window_size, min_timeslice_size)

  start_time = int(features[0][0])
  end_time = int(features[-1][0])

  # Drop the first (timestamp) column.
  features = features[:,1:]

  # Roll the features randomly to give different offsets.
  roll = np.random.randint(0, window_size)
  features = np.roll(features, roll, axis=0)

  if not np.isfinite(features).all():
    logging.fatal('Bad features: %s', features)

  return features, np.array([start_time, end_time], dtype=np.int32)

def np_array_extract_n_features(input, label, n, max_time_delta,
    window_size, min_timeslice_size):
  """ Extract and process multiple timeslices from a vessel movement feature.

  Args:
    input: the input data as a 2d numpy array.
    label: the label for the vessel which made this series.
    n: the number of times to extract a feature timeslice from this series.

  Returns:
    A tuple comprising:
      1. An numpy array comprising N feature timeslices, of dimension
          [n, 1, window_size, num_features].
      2. A numpy array comprising timebounds for each slice, of dimension
          [n, 2].
      3. A numpy array with an int32 label for each slice, of dimension [n].

  """

  samples = []
  for _ in range(n):
    features, time_bounds = np_array_extract_features(input, max_time_delta,
        window_size, min_timeslice_size)
    samples.append((np.stack([features]), time_bounds, label))
  return zip(*samples)

def cropping_weight_replicating_feature_file_reader(filename_queue, num_features,
    max_time_delta, window_size, min_timeslice_size, max_replication_factor):
  """ Set up a file reader and training feature extractor for the files in a queue.

  As a training feature extractor, this pulls sets of random timeslices from the
  vessels found in the files, with the number of draws for each sample determined
  by the weight assigned to the particular vessel.

  Args:
    filename_queue: a queue of filenames for feature files to read.
    num_features: the dimensionality of the features.
    max_replication_factor: the maximum number of samples that can be drawn from a
      single vessel series, regardless of the weight specified.

  Returns:
    A tuple comprising, for the n samples drawn for each vessel:
      1. A tensor of the feature timeslices drawn, of dimension
         [n, 1, window_size, num_features].
      2. A tensor of the timebounds for the timeslices, of dimension [n, 2].
      3. A tensor of the labels for each timeslice, of dimension [n].

  """
  context_features, sequence_features = single_feature_file_reader(filename_queue,
      num_features)

  movement_features = sequence_features['movement_features']
  label = tf.cast(context_features['vessel_type_index'], tf.int32)
  weight = tf.cast(context_features['weight'], tf.float32)

  def replicate_extract(input, label, weight):
    n = int(np.ceil(float(max_replication_factor) * weight))
    return np_array_extract_n_features(input, label, n, max_time_delta,
        window_size, min_timeslice_size)

  features_list, time_bounds_list, label_list = tf.py_func(replicate_extract,
      [movement_features, label, weight], [tf.float32, tf.int32, tf.int32])

  return features_list, time_bounds_list, label_list


