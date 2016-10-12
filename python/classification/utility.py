from collections import defaultdict, namedtuple
import dateutil.parser
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading

VESSEL_CLASS_NAMES = [
    "Purse seine", "Longliner", "Trawler", "Pots and traps", "Squid fishing",
    "Passenger", "Cargo/Tanker", "Seismic", "Tug/Pilot/Supply"
]

VESSEL_CLASS_INDICES = dict(
    zip(VESSEL_CLASS_NAMES, range(len(VESSEL_CLASS_NAMES))))

FishingRange = namedtuple('FishingRange',
                          ['start_time', 'end_time', 'is_fishing'])


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
    timeseries = features[:, 0].astype(np.int32)

    # Roll the features randomly to give different offsets.
    roll = random_state.randint(0, window_size)
    features = np.roll(features, roll, axis=0)

    if not np.isfinite(features).all():
        logging.fatal('Bad features: %s', features)

    return features, timeseries, np.array(
        [start_time, end_time], dtype=np.int32)


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
        features, timeseries, time_bounds = np_array_extract_features(
            random_state, input, max_time_delta, window_size,
            min_timeslice_size)
        samples.append(
            (np.stack([features]), timeseries, time_bounds, np.int32(label)))

    for _ in range(int_n):
        add_sample()

    frac_n = n - float(int_n)
    if (random_state.uniform(0.0, 1.0) <= frac_n):
        add_sample()

    return zip(*samples)


def cropping_weight_replicating_feature_file_reader(
        vessel_metadata, filename_queue, num_features, max_time_delta,
        window_size, min_timeslice_size, max_replication_factor):
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
    mmsi = tf.cast(context_features['mmsi'], tf.int32)
    random_state = np.random.RandomState()

    def replicate_extract(input, mmsi):
        string_label, weight = vessel_metadata[mmsi]
        label = VESSEL_CLASS_INDICES[string_label]
        n = min(float(max_replication_factor), weight)
        return np_array_extract_n_random_features(
            random_state, input, label, n, max_time_delta, window_size,
            min_timeslice_size)

    features_list, timeseries_list, time_bounds_list, label_list = tf.py_func(
        replicate_extract, [movement_features, mmsi],
        [tf.float32, tf.int32, tf.int32, tf.int32])

    return features_list, timeseries_list, time_bounds_list, label_list


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
        cropped = input_series[start_index:end_index]

        # If this window is too long, pick a random subwindow.
        if (length > window_size):
            max_offset = length - window_size
            start_offset = random_state.uniform(max_offset)
            cropped = cropped[max_offset:max_offset + window_size]

        if len(cropped) >= min_points_for_classification:
            output_slice = np_pad_repeat_slice(cropped, window_size)

            time_bounds = np.array([start_time, end_time], dtype=np.int32)

            without_timestamp = output_slice[:, 1:]
            timeseries = output_slice[:, 0].astype(np.int32)
            slices.append(
                (np.stack([without_timestamp]), timeseries, time_bounds, mmsi))

    if slices == []:
        # Return an appropriately shaped empty numpy array.
        return (np.empty(
            [0, 1, window_size, 9], dtype=np.float32), np.empty(
                window_size, dtype=np.int32), np.empty(
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

    features_list, timeseries, time_bounds_list, mmsis = tf.py_func(
        replicate_extract, [movement_features, mmsi],
        [tf.float32, tf.int32, tf.int32, tf.int32])

    return features_list, timeseries, time_bounds_list, mmsis


def read_vessel_metadata_file_lines(available_mmsis, lines):
    """ For a set of vessels, read metadata and calculate class weights.

    Args:
        available_mmsis: a set of all mmsis for which we have feature data.
        lines: a list of comma-separated vessel metadata lines. Columns are
            the mmsi, the split (train/test) and the vessel type
            (Longliner/Passenger etc.).

    Returns:
        A dictionary from data split (training/test) to a dictionary from
        mmsi to the type and weight for a vessel.
    """

    # Build a list of vessels + split + and vessel type. Count the occurrence
    # of each vessel type per split.
    vessel_type_set = set()
    dataset_kind_counts = defaultdict(lambda: defaultdict(lambda: 0))
    vessel_types = []
    for line in lines[1:]:
        mmsi_str, split, vessel_type = line.strip().split(',')
        mmsi = int(mmsi_str)
        if mmsi in available_mmsis:
            vessel_types.append((mmsi, split, vessel_type))
            dataset_kind_counts[split][vessel_type] += 1
            vessel_type_set.add(vessel_type)

    # Calculate weights for each vessel type per split: the weight is the count
    # of the most frequent vessel type divided by the count for the current
    # vessel type. Used to sample more frequently from less-represented vessel
    # types.
    dataset_kind_weights = defaultdict(lambda: {})
    for split, counts in dataset_kind_counts.iteritems():
        max_count = max(counts.values())
        for vessel_type, count in counts.iteritems():
            dataset_kind_weights[split][vessel_type] = float(
                max_count) / float(count)

    metadata_dict = defaultdict(lambda: {})
    for mmsi, split, vessel_type in vessel_types:
        metadata_dict[split][mmsi] = (vessel_type,
                                      dataset_kind_weights[split][vessel_type])

    if len(vessel_type_set) == 0:
        logging.fatal('No vessel types found for training.')
        sys.exit(-1)

    logging.info("Vessel types: %s", list(vessel_type_set))

    return metadata_dict


def read_vessel_metadata(available_mmsis, metadata_file):
    with open(metadata_file, 'r') as f:
        return read_vessel_metadata_file_lines(available_mmsis, f.readlines())


def read_fishing_ranges(fishing_range_file):
    fishing_range_dict = defaultdict(lambda: [])
    with open(fishing_range_file, 'r') as f:
        for l in f.readlines()[1:]:
            els = l.split(',')
            mmsi = int(els[0])
            start_time = dateutil.parser.parse(els[1])
            end_time = dateutil.parser.parse(els[2])
            is_fishing = float(els[3])

            fishing_range_dict[mmsi].append(
                FishingRange(start_time, end_time, is_fishing))
    return fishing_range_dict
