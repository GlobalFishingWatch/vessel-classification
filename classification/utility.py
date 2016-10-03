import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading

VESSEL_CLASS_NAMES = [
    "Unknown", "Purse seine", "Longliner", "Trawler", "Pots and traps",
    "Passenger", "Tug", "Cargo/Tanker", "Supply"
]


class ModelConfiguration(object):
    """ Configuration for the vessel behaviour model, shared between training and
      inference.
  """

    def __init__(self):
        self.feature_duration_days = 180
        self.num_classes = 9
        self.num_feature_dimensions = 9
        self.max_sample_frequency_seconds = 5 * 60
        self.max_window_duration_seconds = self.feature_duration_days * 24 * 3600

        # We allocate a much smaller buffer than would fit the specified time
        # sampled at 5 mins intervals, on the basis that the sample is almost
        # always much more sparse.
        self.window_max_points = (self.max_window_duration_seconds /
                                  self.max_sample_frequency_seconds) / 4
        self.window_size = 3
        self.stride = 2
        self.feature_depth = 20
        self.levels = 10
        self.batch_size = 32
        self.min_viable_timeslice_length = 500

    def zero_pad_features(self, features):
        """ Zero-pad features in the depth dimension to match requested feature depth. """

        feature_pad_size = self.feature_depth - self.num_feature_dimensions
        assert (feature_pad_size >= 0)
        zero_padding = tf.zeros([self.batch_size, 1, self.window_max_points,
                                 feature_pad_size])
        padded = tf.concat(3, [features, zero_padding])

        return padded


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
        return self.task_index == 0

    def create_server(self):
        server = tf.train.Server(
            self.cluster_spec,
            job_name=self.task_type,
            task_index=self.task_index)

        logging.info("Server target: %s", server.target)
        return server

    @staticmethod
    def create_local_server_config():
        return ClusterNodeConfig({"cluster": {},
                                  "task": {"type": "worker",
                                           "index": 0}})


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
        context_features={
            'mmsi': tf.FixedLenFeature([], tf.int64),
            'weight': tf.FixedLenFeature([], tf.float32),
            'vessel_type_index': tf.FixedLenFeature([], tf.int64),
            'vessel_type_name': tf.FixedLenFeature([], tf.string)
        },
        sequence_features={
            'movement_features': tf.FixedLenSequenceFeature(
                shape=(num_features, ), dtype=tf.float32)
        })

    return context_features, sequence_features


def np_pad_repeat_slice(slice, window_size):
    """ Pads slice to the specified window size.

  Series that are shorter than window_size are repeated into unfilled space.

  Args:
    slice: a numpy array.
    window_size: the size the array must be padded to.

  Returns:
    a numpy array of length window_size in the first dimension.
  """

    slice_length = len(slice)
    assert (slice_length <= window_size)
    reps = int(np.ceil(window_size / float(slice_length)))
    return np.concatenate([slice] * reps, axis=0)[:window_size]


def np_array_random_fixed_time_extract(random_state, input_series,
                                       max_time_delta, output_length,
                                       min_timeslice_size):
    """ Extracts a random fixed-time slice from a 2d numpy array.
    
  The input array must be 2d, representing a time series, with the first    
  column representing a timestamp (sorted ascending). Any values in the series    
  with a time greater than (first time + max_time_delta) are removed and the    
  prefix series repeated into the window to pad.    
    
  Args:
    random_state: a numpy randomstate object.
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
        time_offset = random_state.randint(0, max_time_offset)
    start_index = np.searchsorted(
        input_series[:, 0], start_time + time_offset, side='left')

    # Should not start closer than min_timeslice_size points from the end lest the 
    # series have too few points to be meaningful.
    start_index = min(start_index, max(0, input_length - min_timeslice_size))
    crop_end_time = min(input_series[start_index][0] + max_time_delta,
                        end_time)

    end_index = min(start_index + output_length,
                    np.searchsorted(
                        input_series[:, 0], crop_end_time, side='right'))

    cropped = input_series[start_index:end_index]
    output_series = np_pad_repeat_slice(cropped, output_length)

    return output_series


def np_array_extract_features(random_state, input, max_time_delta, window_size,
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
        random_state, input, max_time_delta, window_size, min_timeslice_size)

    start_time = int(features[0][0])
    end_time = int(features[-1][0])

    # Drop the first (timestamp) column.
    features = features[:, 1:]

    # Roll the features randomly to give different offsets.
    roll = random_state.randint(0, window_size)
    features = np.roll(features, roll, axis=0)

    if not np.isfinite(features).all():
        logging.fatal('Bad features: %s', features)

    return features, np.array([start_time, end_time], dtype=np.int32)


def np_array_extract_n_random_features(random_state, input, label, n,
                                       max_time_delta, window_size,
                                       min_timeslice_size):
    """ Extract and process multiple random timeslices from a vessel movement feature.

  Args:
    input: the input data as a 2d numpy array.
    label: the label for the vessel which made this series.
    n: the (floating-point) number of times to extract a feature timeslice from
       this series.

  Returns:
    A tuple comprising:
      1. An numpy array comprising N feature timeslices, of dimension
          [n, 1, window_size, num_features].
      2. A numpy array comprising timebounds for each slice, of dimension
          [n, 2].
      3. A numpy array with an int32 label for each slice, of dimension [n].

  """

    samples = []
    int_n = int(n)

    def add_sample():
        features, time_bounds = np_array_extract_features(
            random_state, input, max_time_delta, window_size,
            min_timeslice_size)
        samples.append((np.stack([features]), time_bounds, label))

    for _ in range(int_n):
        add_sample()

    frac_n = n - float(int_n)
    if (random_state.uniform(0.0, 1.0) <= frac_n):
        add_sample()

    return zip(*samples)


def cropping_weight_replicating_feature_file_reader(
        filename_queue, num_features, max_time_delta, window_size,
        min_timeslice_size, max_replication_factor):
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
    context_features, sequence_features = single_feature_file_reader(
        filename_queue, num_features)

    movement_features = sequence_features['movement_features']
    label = tf.cast(context_features['vessel_type_index'], tf.int32)
    weight = tf.cast(context_features['weight'], tf.float32)

    random_state = np.random.RandomState()

    def replicate_extract(input, label, weight):
        n = min(float(max_replication_factor), weight)
        return np_array_extract_n_random_features(
            random_state, input, label, n, max_time_delta, window_size,
            min_timeslice_size)

    features_list, time_bounds_list, label_list = tf.py_func(
        replicate_extract, [movement_features, label, weight],
        [tf.float32, tf.int32, tf.int32])

    return features_list, time_bounds_list, label_list


def np_array_extract_slices_for_time_ranges(random_state, input_series, mmsi,
                                            time_ranges, window_size,
                                            min_points_for_classification):
    """ Extract and process a set of specified time slices from a vessel
        movement feature.

    Args:
        random_state: a numpy randomstate object.
        input: the input data as a 2d numpy array.
        mmsi: the id of the vessel which made this series.
        max_time_delta: the maximum time contained in each window.
        window_size: the size of the window.
        min_points_for_classification: the minumum number of points in a window for
            it to be usable.

    Returns:
      A tuple comprising:
        1. An numpy array comprising N feature timeslices, of dimension
            [n, 1, window_size, num_features].
        2. A numpy array comprising timebounds for each slice, of dimension
            [n, 2].
        3. A numpy array with an int32 mmsi for each slice, of dimension [n].

    """
    slices = []
    times = input_series[:, 0]
    for (start_time, end_time) in time_ranges:
        start_index = np.searchsorted(times, start_time, side='left')
        end_index = np.searchsorted(times, end_time, side='left')
        length = end_index - start_index

        # Slice out the time window, removing the timestamp.
        cropped = input_series[start_index:end_index, 1:]

        # If this window is too long, pick a random subwindow.
        if (length > window_size):
            max_offset = length - window_size
            start_offset = random_state.uniform(max_offset)
            cropped = cropped[max_offset:max_offset + window_size]

        if len(cropped) >= min_points_for_classification:
            output_slice = np_pad_repeat_slice(cropped, window_size)

            time_bounds = np.array([start_time, end_time], dtype=np.int32)
            slices.append((np.stack([output_slice]), time_bounds, mmsi))

    if slices == []:
        # Return an appropriately shaped empty numpy array.
        return (np.empty(
            [0, 1, window_size, 9], dtype=np.float32), np.empty(
                shape=[0, 2], dtype=np.int32), np.empty(
                    shape=[0], dtype=np.int32))

    return zip(*slices)


def cropping_all_slice_feature_file_reader(filename_queue, num_features,
                                           time_ranges, window_size,
                                           min_points_for_classification):
    """ Set up a file reader and inference feature extractor for the files in a queue.

  An inference feature extractor, pulling all sequential slices from a vessel
  movement series.

  Args:
    filename_queue: a queue of filenames for feature files to read.
    num_features: the dimensionality of the features.

  Returns:
    A tuple comprising, for the n slices comprising each vessel:
      1. A tensor of the feature timeslices drawn, of dimension
         [n, 1, window_size, num_features].
      2. A tensor of the timebounds for the timeslices, of dimension [n, 2].
      3. A tensor of the labels for each timeslice, of dimension [n].

  """
    context_features, sequence_features = single_feature_file_reader(
        filename_queue, num_features)

    movement_features = sequence_features['movement_features']
    mmsi = tf.cast(context_features['mmsi'], tf.int32)

    random_state = np.random.RandomState()

    def replicate_extract(input, mmsi):
        return np_array_extract_slices_for_time_ranges(
            random_state, input, mmsi, time_ranges, window_size,
            min_points_for_classification)

    features_list, time_bounds_list, mmsis = tf.py_func(
        replicate_extract, [movement_features, mmsi],
        [tf.float32, tf.int32, tf.int32])

    return features_list, time_bounds_list, mmsis
