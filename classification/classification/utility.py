from collections import defaultdict, namedtuple
import csv
import dateutil.parser
import hashlib
import math
import model
import time
import logging
import newlinejson as nlj
import numpy as np
import os
import struct
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
""" The main column for vessel classification. """
PRIMARY_VESSEL_CLASS_COLUMN = 'label'
""" The coarse vessel label set. """
VESSEL_CLASS_NAMES = ['Passenger', 'Squid', 'Cargo/Tanker', 'Trawlers',
                      'Seismic vessel', 'Set gillnets', 'Longliners', 'Reefer',
                      'Pole and Line', 'Purse seines', 'Pots and Traps',
                      'Trollers', 'Tug/Pilot/Supply']
""" The finer vessel label set. """
VESSEL_CLASS_DETAILED_NAMES = [
    'Squid', 'Trawlers', 'Seismic vessel', 'Set gillnets', 'Reefer',
    'Pole and Line', 'Purse seines', 'Pots and Traps', 'Trollers', 'Cargo',
    'Sailing', 'Supply', 'Set longlines', 'Motor Passenger',
    'Drifting longlines', 'Tanker', 'Tug', 'Pilot'
]
""" The vessel length classes. """
VESSEL_LENGTH_CLASSES = [
    '0-12m', '12-18m', '18-24m', '24-36m', '36-50m', '50-75m', '75-100m',
    '100-150m', '150m+'
]

TEST_SPLIT = 'Test'
TRAINING_SPLIT = 'Training'


def vessel_categorical_length_transformer(vessel_length_string):
    """ A transformer from continuous vessel lengths to discrete categories. """
    ranges = [12.0, 18.0, 24.0, 36.0, 50.0, 75.0, 100.0, 150.0]

    if vessel_length_string == '':
        return ''

    vessel_length = float(vessel_length_string)
    for i in range(len(ranges)):
        if vessel_length < ranges[i]:
            return VESSEL_LENGTH_CLASSES[i]
    return VESSEL_LENGTH_CLASSES[-1]


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
        return ClusterNodeConfig({
            "cluster": {},
            "task": {
                "type": "master",
                "index": 0
            }
        })


def fishing_localisation_loss(logits, targets):
    """A loss function for fishing localisation, which takes into account the
       fact that we frequently do not have information about when fishing is
       happening. Thus targets can be in the range 0 (not fishing) - 1 (fishing)
       or it can take the value -1 to indicate don't know.
    """
    cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)

    mask = tf.select(
        tf.equal(targets, -1),
        tf.zeros_like(
            targets, dtype=tf.float32),
        tf.ones_like(
            targets, dtype=tf.float32))
    input_size = tf.cast(tf.gather(tf.shape(logits), 1), tf.float32)

    # Scale the sum of the loss by the number of present points in the
    # target data, or 10% of the window size: whichever is the larger. Do this
    # per sample (not across the batches).
    loss_scale = tf.maximum(
        tf.reduce_sum(
            mask, reduction_indices=[1]), 0.1 * input_size)

    return tf.reduce_mean(
        tf.reduce_sum(
            cross_entropies * mask, reduction_indices=[1]) / loss_scale)


def fishing_localisation_mse(predictions, targets):
    """Mean squared error for fishing localisation, which takes into account the
       fact that we frequently do not have information about when fishing is
       happening. Thus targets can be in the range 0 (not fishing) - 1 (fishing)
       or it can take the value -1 to indicate don't know.
    """
    mask = tf.select(
        tf.equal(targets, -1),
        tf.zeros_like(
            targets, dtype=tf.float32),
        tf.ones_like(
            targets, dtype=tf.float32))
    scale = tf.reduce_sum(mask)

    error = (predictions - targets) * mask
    mse_sum = tf.reduce_sum(error * error)

    return tf.cond(
        tf.equal(scale, 0.0), lambda: scale, lambda: mse_sum / scale)


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
        context_features={'mmsi': tf.FixedLenFeature([], tf.int64), },
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
      2. The timestamps of each feature point.
      3. The start and end time of the timeslice (in int32 seconds since epoch).
  """
    features = np_array_random_fixed_time_extract(
        random_state, input, max_time_delta, window_size, min_timeslice_size)

    start_time = int(features[0][0])
    end_time = int(features[-1][0])

    # Roll the features randomly to give different offsets.
    roll = random_state.randint(0, window_size)
    features = np.roll(features, roll, axis=0)

    # Drop the first (timestamp) column.
    timestamps = features[:, 0].astype(np.int32)
    features = features[:, 1:]

    if not np.isfinite(features).all():
        logging.fatal('Bad features: %s', features)

    return features, timestamps, np.array(
        [start_time, end_time], dtype=np.int32)


def np_array_extract_n_random_features(random_state, input, n, max_time_delta,
                                       window_size, min_timeslice_size, mmsi):
    """ Extract and process multiple random timeslices from a vessel movement feature.

  Args:
    input: the input data as a 2d numpy array.
    training_labels: the label for the vessel which made this series.
    n: the (floating-point) number of times to extract a feature timeslice from
       this series.

  Returns:
    A tuple comprising:
      1. N extracted feature timeslices.
      2. N lists of timestamps for each feature point.
      3. N start and end times for each the timeslice.
      4. N mmsis, one per feature slice.
  """

    samples = []
    int_n = int(n)

    def add_sample():
        features, timestamps, time_bounds = np_array_extract_features(
            random_state, input, max_time_delta, window_size,
            min_timeslice_size)

        samples.append((np.stack([features]), timestamps, time_bounds, mmsi))

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
        fishing_ranges: a dictionary of fishing ranges per mmsi.

    Returns:
        A tuple comprising, for the n samples drawn for each vessel:
            1. A tensor of the feature timeslices drawn, of dimension
               [n, 1, window_size, num_features].
            2. A tensor of the timebounds for the timeslices, of dimension [n, 2].
            3. A tensor of the labels for each timeslice, of dimension [n].
            4. A tensor of the mmsis for each timeslice, of dimension [n].
    """
    context_features, sequence_features = single_feature_file_reader(
        filename_queue, num_features)

    movement_features = sequence_features['movement_features']
    mmsi = tf.cast(context_features['mmsi'], tf.int32)
    random_state = np.random.RandomState()

    def replicate_extract(input, mmsi):
        n = 4.0
        return np_array_extract_n_random_features(random_state, input, n,
                                                  max_time_delta, window_size,
                                                  min_timeslice_size, mmsi)

    (features_list, timestamps, time_bounds_list, mmsis) = tf.py_func(
        replicate_extract, [movement_features, mmsi],
        [tf.float32, tf.int32, tf.int32, tf.int32])

    return features_list, timestamps, time_bounds_list, mmsis


def np_array_extract_slices_for_time_ranges(
        random_state, input_series, num_features_inc_timestamp, mmsi,
        time_ranges, window_size, min_points_for_classification):
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
        2. A numpy array with the timestamps for each feature point, of
           dimension [n, window_size].
        3. A numpy array comprising timebounds for each slice, of dimension
            [n, 2].
        4. A numpy array with an int32 mmsi for each slice, of dimension [n].

    """
    slices = []
    times = input_series[:, 0]
    for (start_time, end_time) in time_ranges:
        start_index = np.searchsorted(times, start_time, side='left')
        end_index = np.searchsorted(times, end_time, side='left')
        length = end_index - start_index

        # Slice out the time window.
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
            [0, 1, window_size, num_features_inc_timestamp - 1],
            dtype=np.float32), np.empty(
                shape=[0, window_size], dtype=np.int32), np.empty(
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
          2. A tensor of the timestamps for each feature point of dimension
             [n, window_size].
          3. A tensor of the timebounds for the timeslices, of dimension [n, 2].
          4. A tensor of the mmsis of each vessel of dimension [n].

    """
    context_features, sequence_features = single_feature_file_reader(
        filename_queue, num_features)

    movement_features = sequence_features['movement_features']
    mmsi = tf.cast(context_features['mmsi'], tf.int32)

    random_state = np.random.RandomState()

    def replicate_extract(input, mmsi):
        return np_array_extract_slices_for_time_ranges(
            random_state, input, num_features, mmsi, time_ranges, window_size,
            min_points_for_classification)

    features_list, timeseries, time_bounds_list, mmsis = tf.py_func(
        replicate_extract, [movement_features, mmsi],
        [tf.float32, tf.int32, tf.int32, tf.int32])

    return features_list, timeseries, time_bounds_list, mmsis


def _hash_mmsi_to_double(mmsi, salt=''):
    """Take a value and hash it to return a value in the range [0, 1.0).
     To be used as a deterministic probability for vessel dataset
     assignment: e.g. if we decide vessels should go in the training set at
     probability 0.2, then we map from mmsi to a probability, then if the value
     is <= 0.2 we assign this vessel to the training set.
    Args:
        mmsi: the input MMSI as an integer.
        salt: a salt concatenated to the mmsi to allow more than one value to be
                    generated per mmsi.
    Returns:
        A value in the range [0, 1.0).
    """
    hasher = hashlib.md5()
    i = '%s_%s' % (mmsi, salt)
    hasher.update(i)

    # Pick a number of bytes from the bottom of the hash, and scale the value
    # by the max value that an unsigned integer of that size can have, to get a
    # value in the range [0, 1.0)
    hash_bytes_for_value = 4
    hash_value = struct.unpack('I', hasher.digest()[:hash_bytes_for_value])[0]
    sample = float(hash_value) / math.pow(2.0, hash_bytes_for_value * 8)
    assert sample >= 0.0
    assert sample <= 1.0
    return sample


class VesselMetadata(object):
    def __init__(self,
                 metadata_dict,
                 fishing_ranges_map,
                 fishing_range_training_upweight=1.0):
        self.metadata_by_split = metadata_dict
        self.metadata_by_mmsi = {}
        self.fishing_ranges_map = fishing_ranges_map
        self.fishing_range_training_upweight = fishing_range_training_upweight
        for split, vessels in metadata_dict.iteritems():
            for mmsi, data in vessels.iteritems():
                self.metadata_by_mmsi[mmsi] = data

        intersection_mmsis = set(self.metadata_by_mmsi.keys()).intersection(
            set(fishing_ranges_map.keys()))
        logging.info("Metadata for %d mmsis." % len(self.metadata_by_mmsi))
        logging.info("Fishing ranges for %d mmsis." % len(fishing_ranges_map))
        logging.info("Vessels with both types of data: %d",
                     len(intersection_mmsis))

    def vessel_weight(self, mmsi):
        if mmsi in self.fishing_ranges_map:
            fishing_range_multiplier = self.fishing_range_training_upweight
        else:
            fishing_range_multiplier = 1.0

        return self.metadata_by_mmsi[mmsi][1] * fishing_range_multiplier

    def vessel_label(self, label_name, mmsi):
        return self.metadata_by_mmsi[mmsi][0][label_name]

    def mmsis_for_split(self, split):
        return self.metadata_by_split[split].keys()

    def weighted_training_list(self, random_state, split,
                               max_replication_factor):
        replicated_mmsis = []
        logging.info("Training mmsis: %d",
                     len(self.metadata_by_split[split].keys()))
        fishing_ranges_mmsis = []
        for mmsi, (rows, weight) in self.metadata_by_split[split].iteritems():

            if mmsi in self.fishing_ranges_map:
                fishing_ranges_mmsis.append(mmsi)
                weight = weight * self.fishing_range_training_upweight

            weight = min(weight, max_replication_factor)

            int_n = int(weight)
            replicated_mmsis = replicated_mmsis + ([mmsi] * int_n)
            frac_n = weight - float(int_n)
            if (random_state.uniform(0.0, 1.0) <= frac_n):
                replicated_mmsis.append(mmsi)

        random_state.shuffle(replicated_mmsis)
        logging.info("Replicated training mmsis: %d", len(replicated_mmsis))
        logging.info("Fishing range mmsis: %d", len(fishing_ranges_mmsis))

        return replicated_mmsis


def read_vessel_multiclass_metadata_lines(available_mmsis, lines,
                                          fishing_range_dict,
                                          fishing_range_training_upweight):
    """ For a set of vessels, read metadata and calculate class weights.

    Args:
        available_mmsis: a set of all mmsis for which we have feature data.
        lines: a list of comma-separated vessel metadata lines. Columns are
            the mmsi and a set of vessel type columns, containing at least one
            called 'label' being the primary/coarse type of the vessel e.g.
            (Longliner/Passenger etc.).

    Returns:
        A VesselMetadata object with weights and labels for each vessel.
    """

    vessel_type_set = set()
    dataset_kind_counts = defaultdict(lambda: defaultdict(lambda: 0))
    vessel_types = []

    # Build a list of vessels + split + and vessel type. Calculate the split on
    # the fly, but deterministically. Count the occurrence of each vessel type
    # per split.
    for row in lines:
        mmsi = int(row['mmsi'])
        coarse_vessel_type = row[PRIMARY_VESSEL_CLASS_COLUMN]
        if mmsi in available_mmsis and coarse_vessel_type:
            if (_hash_mmsi_to_double(mmsi) >= 0.5):
                split = 'Test'
            else:
                split = 'Training'
            vessel_types.append((mmsi, split, coarse_vessel_type, row))
            dataset_kind_counts[split][coarse_vessel_type] += 1
            vessel_type_set.add(coarse_vessel_type)

    # Calculate weights for each vessel type per split: the weight is the count
    # of the most frequent vessel type divided by the count for the current
    # vessel type. Used to sample more frequently from less-represented vessel
    # types.
    dataset_kind_weights = defaultdict(lambda: {})
    for split, counts in dataset_kind_counts.iteritems():
        max_count = max(counts.values())
        for coarse_vessel_type, count in counts.iteritems():
            dataset_kind_weights[split][coarse_vessel_type] = float(
                max_count) / float(count)

    metadata_dict = defaultdict(lambda: {})
    for mmsi, split, coarse_vessel_type, row in vessel_types:
        metadata_dict[split][mmsi] = (
            row, dataset_kind_weights[split][coarse_vessel_type])

    if len(vessel_type_set) == 0:
        logging.fatal('No vessel types found for training.')
        sys.exit(-1)

    logging.info("Vessel types: %s", list(vessel_type_set))

    return VesselMetadata(metadata_dict, fishing_range_dict,
                          fishing_range_training_upweight)


def read_vessel_multiclass_metadata(available_mmsis,
                                    metadata_file,
                                    fishing_range_dict={},
                                    fishing_range_training_upweight=1.0):
    with open(metadata_file, 'r') as f:
        reader = csv.DictReader(f)
        logging.info("Metadata columns: %s", reader.fieldnames)
        return read_vessel_multiclass_metadata_lines(
            available_mmsis, reader, fishing_range_dict,
            fishing_range_training_upweight)


def find_available_mmsis(feature_path):
    # TODO(alexwilson): Using a temporary session to get the matching files on
    # GCS is far from ideal. However the alternative is to bring in additional
    # libraries with explicit auth that may or may not play nicely with CloudML.
    # Improve later...
    mmsi_cache_filename = "available_mmsis.cache"
    if os.path.exists(mmsi_cache_filename):
        with nlj.open(mmsi_cache_filename, 'r') as cache:
            for line in cache:
                if line["path"] == feature_path:
                    logging.info("Loading mmsis from cache.")
                    return set(line['mmsis'])

    with tf.Session() as sess:
        logging.info(
            "Finding matching features files. May take a few minutes...")
        matching_files = tf.train.match_filenames_once(feature_path +
                                                       "/*.tfrecord")
        sess.run(tf.initialize_all_variables())

        all_feature_files = sess.run(matching_files)
        if len(all_feature_files) == 0:
            logging.fatal("Error: no feature files found.")
            sys.exit(-1)
        logging.info("Found %d feature files.", len(all_feature_files))

    mmsis = [int(os.path.split(p)[1].split('.')[0]) for p in all_feature_files]
    with nlj.open(mmsi_cache_filename, 'a') as cache:
        cache.write({"path": feature_path, "mmsis": mmsis})

    return set(mmsis)


def read_fishing_ranges(fishing_range_file):
    """ Read vessel fishing ranges, return a dict of mmsi to classified fishing
        or non-fishing ranges for that vessel.
    """
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
