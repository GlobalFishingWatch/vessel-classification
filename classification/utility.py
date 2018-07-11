# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, namedtuple
import csv
import datetime
import dateutil.parser
import pytz
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
import yaml

# Upweight false positives to strongly discourage transits
FALSE_POSITIVE_UPWEIGHT = 1
MAX_UPWEIGHT = 100
""" The main column for vessel classification. """
PRIMARY_VESSEL_CLASS_COLUMN = 'label'

#TODO: (bitsofbits) think about extracting to config file

# The 'real' categories for multihotness are the fine categories, which 'coarse' and 'fishing' 
# are defined in terms of. Any number of coarse categories, even with overlapping values can 
# be defined in principle, although at present the interaction between the mulithot and non multihot
# versions makes that more complicated.

schema = yaml.load('''
unknown:
  non_fishing:
    passenger:
    gear:
    fish_factory:
    cargo_or_tanker:
      bunker_or_tanker:
        bunker:
        tanker:
      cargo_or_reefer:
        cargo:
        reefer:
          specialized_reefer:
          container_reefer:
        fish_tender:
          well_boat:
    patrol_vessel:
    research:
    dive_vessel:
    submarine:
    dredge_non_fishing:
    supply_vessel:
    tug:
    seismic_vessel:
    helicopter:
    other_not_fishing:

  fishing:
    squid_jigger:
    drifting_longlines:
    pole_and_line:
    other_fishing:
    trollers:
    fixed_gear:
      pots_and_traps:
      set_longlines:
      set_gillnets:
    trawlers:
    dredge_fishing:
    seiners:
     purse_seines:
      tuna_purse_seines:
      other_purse_seines:
     other_seines:
    driftnets:
''')


def atomic(obj):
    for k, v in obj.items():
        if v is None:
            yield k
        else:
            for x in atomic(v):
                yield x

def categories(obj, include_atomic=True):
    for k, v in obj.items():
        if v is None:
            if include_atomic:
                yield k, [k]
        else:
            yield (k, list(atomic(v)))
            for x in categories(v, include_atomic=include_atomic):
                yield x




#TODO: Better names
VESSEL_CLASS_DETAILED_NAMES = sorted(atomic(schema))

VESSEL_CATEGORIES = sorted(categories(schema))

TEST_SPLIT = 'Test'
TRAINING_SPLIT = 'Training'


def repeat_tensor(input, n):
    batch_size, _, width, depth = input.get_shape()
    repeated = tf.concat([input] * n, 3)
    return tf.reshape(repeated, [-1, 1, int(width) * n, int(depth)])


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
                "type": "worker",
                "index": 0
            }
        })


def fishing_localisation_loss(logits, targets):
    """A loss function for fishing localisation, which takes into account the
       fact that we frequently do not have information about when fishing is
       happening. Thus targets can be in the range 0 (not fishing) - 1 (fishing)
       or it can take the value -1 to indicate don't know.
    """
    cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)

    mask = tf.where(
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
    EPSILON = 1e-10
    mask = tf.to_float(tf.not_equal(targets, -1))
    scale = tf.reduce_sum(mask)

    error = (predictions - targets) * mask
    mse_sum = tf.reduce_sum(error * error)

    return mse_sum / (scale + EPSILON)


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


def np_array_random_fixed_length_extract(random_state, input_series,
                                         output_length):

    cropped = input_series[start_offset:end_offset]

    return np_pad_repeat_slice(cropped, output_length)


def np_array_random_fixed_points_extract(random_state, input_series,
                                         output_length, min_timeslice_size,
                                         selection_ranges, mmsi):
    """ Extracts a random fixed-points slice from a 2d numpy array.
    
    The input array must be 2d, representing a time series, with the first    
    column representing a timestamp (sorted ascending). 

    Args:
        random_state: a numpy randomstate object.
        input_series: the input series. A 2d array first column representing an
            ascending time.   
        output_length: the number of points in the output series. Input series    
            shorter than this will be repeated into the output series.   
        min_timeslice_size: the minimum number of points in a timeslice for the
            series to be considered meaningful. 
        selection_ranges: Either a list of time ranges that should be preferentially 
            selected from (we try to get at least on point from one of the ranges), or 
            None if to disable this behaviour. 

    Returns:    
        An array of the same depth as the input, but altered width, representing
        the fixed points slice.   
    """

    input_length = len(input_series)

    def extract_start_end(min_index, max_index):
        max_start = min(max_index, input_length - output_length)
        if max_start >= min_index:
            ndx = random_state.randint(min_index, max_start)
            return ndx, ndx + output_length
        else:
            logging.warning('Cant grab data for range for %s (%s %s)',
                      mmsi, input_length, output_length)
            return None, None

    if selection_ranges:

        # Copy and shuffle the ranges so we see them in a random order
        selections_ranges = list(selection_ranges)
        random_state.shuffle(selections_ranges)

        for sel_range in selections_ranges:
            # For each range figure out the min and max acceptable point in input_series
            # If there are any points then pick a range that either: 
            # a. if there <= output_length points, include all of them
            # b. if there are > output_length points include a random subset.
            rng_start_stamp = (sel_range.start_time - datetime.datetime(
                1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
            rng_start_ndx = np.searchsorted(input_series[:, 0],
                                            rng_start_stamp, side='left')

            rng_end_stamp = (sel_range.end_time - datetime.datetime(
                1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
            rng_end_ndx = np.searchsorted(input_series[:, 0], rng_end_stamp, side='right')

            if rng_end_ndx <= rng_start_ndx:
                continue

            min_ndx = max(rng_start_ndx - output_length + 1,
                          0)  

            max_ndx = min(rng_end_ndx + output_length - 1 + 1,
                          input_length - 1) 

            max_start = min(max_ndx, input_length - output_length)

            if max_start >= min_ndx:
                start_index = random_state.randint(min_ndx, max_start)
                end_index = start_index + output_length
                break
        else:
            logging.warning('Pulling data for %s from full range (input_length = %s)',
                      mmsi, input_length)
            start_index, end_index = extract_start_end(0, input_length - 1)
 
    cropped = input_series[start_index:end_index]
    output_series = np_pad_repeat_slice(cropped, output_length)

    return output_series


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
    assert max_time_delta != 0, 'max_time_delta must be non zero for time based windows'

    # We want to include min_timeslice_size points in our data if we can, so we try picking
    # slices `TRIALS` time.
    TRIALS = 128

    for _ in range(TRIALS):
        input_length = len(input_series)

        start_time = input_series[0][0]
        end_time = input_series[-1][0]
        max_time_offset = max((end_time - start_time) - max_time_delta, 0)
        time_offset = random_state.randint(0, max_time_offset + 1)

        start_index = np.searchsorted(
            input_series[:, 0], start_time + time_offset, side='left')

        # Should not start closer than min_timeslice_size points from the end lest the 
        # series have too few points to be meaningful.
        start_index = min(start_index, max(0,
                                           input_length - min_timeslice_size))
        crop_end_time = min(input_series[start_index][0] + max_time_delta,
                            end_time)

        end_index = min(start_index + output_length,
                        np.searchsorted(
                            input_series[:, 0], crop_end_time, side='right'))

        cropped = input_series[start_index:end_index]

        if len(cropped) >= min_timeslice_size:
            # Use this try
            break

        # Otherwise try again unless we are out of trials.

    return np_pad_repeat_slice(cropped, output_length)


def np_array_extract_features(random_state, input, max_time_delta, window_size,
                              min_timeslice_size, selection_ranges, mmsi):
    """ Extract and process a random timeslice from vessel movement features.

  Removes the timestamp column from the features, and applies a random roll to
  the chosen timeslice to further augment the training data.

  Args:
    random_state: a numpy RandomState object.
    input: the input data as a 2d numpy array.
    max_time_delta: the maximum duration of the returned timeseries in seconds.
    window_size: the number of points in the window
    min_timeslice_size: the minimum number of points in a timeslice for the
                        series to be considered meaningful.
    selection_ranges: Either a list of timeranges that should be preferentially selected 
                      from (we try to get at least on point from one of the ranges), or None if
                      to disable this behaviour. 

    Note that if max_time_delta must be zero, or selection_ranges must be None, but not
    both.

  Returns:
    A tuple comprising:
      1. The extracted feature timeslice.
      2. The timestamps of each feature point.
      3. The start and end time of the timeslice (in int32 seconds since epoch).
  """

    if max_time_delta == 0:
        features = np_array_random_fixed_points_extract(
            random_state, input, window_size, min_timeslice_size,
            selection_ranges, mmsi)
    else:
        features = np_array_random_fixed_time_extract(
            random_state, input, max_time_delta, window_size,
            min_timeslice_size)

    start_time = int(features[0][0])
    end_time = int(features[-1][0])

    # Drop the first (timestamp) column.
    timestamps = features[:, 0].astype(np.int32)
    features = features[:, 1:]

    if not np.isfinite(features).all():
        logging.fatal('Bad features: %s', features)

    return features, timestamps, np.array(
        [start_time, end_time], dtype=np.int32)


def np_array_extract_n_random_features(random_state, input, n, max_time_delta,
                                       window_size, min_timeslice_size, mmsi,
                                       selection_ranges):
    """ Extract and process multiple random timeslices from a vessel movement feature.

  Args:
    input: the input data as a 2d numpy array.
    training_labels: the label for the vessel which made this series.
    n: the number of times to extract a feature timeslice from
       this series.

  Returns:
    A tuple comprising:
      1. N extracted feature timeslices.
      2. N lists of timestamps for each feature point.
      3. N start and end times for each the timeslice.
      4. N mmsis, one per feature slice.
  """

    samples = []

    for _ in range(n):
        features, timestamps, time_bounds = np_array_extract_features(
            random_state, input, max_time_delta, window_size,
            min_timeslice_size, selection_ranges, mmsi)

        samples.append((np.stack([features]), timestamps, time_bounds, mmsi))

    return zip(*samples)


def random_feature_cropping_file_reader(vessel_metadata,
                                        filename_queue,
                                        num_features,
                                        max_time_delta,
                                        window_size,
                                        min_timeslice_size,
                                        select_ranges=False):
    """ Set up a file reader and training feature extractor for the files in a queue.

    As a training feature extractor, this pulls sets of random timeslices from the
    vessels found in the files, with the number of draws for each sample determined
    by the weight assigned to the particular vessel.

    Args:
        vessel_metadata: VesselMetadata object
        filename_queue: a queue of filenames for feature files to read.
        max_time_delta: the maximum duration of the returned timeseries in seconds.
        window_size: the number of points in the window
        min_timeslice_size: the minimum number of points in a timeslice for the
                            series to be considered meaningful.
        select_ranges: bool; should we choose ranges based on fishing_range_map
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
    int_mmsi = tf.cast(context_features['mmsi'], tf.int64)
    random_state = np.random.RandomState()

    num_slices_per_mmsi = 8

    def replicate_extract(input, int_mmsi):
        # Extract several random windows from each vessel track
        # TODO: Fix feature generation so it returns strings directly
        mmsi = vessel_metadata.mmsi_map_int2str[int_mmsi]
        if mmsi in vessel_metadata.fishing_ranges_map:
            ranges = vessel_metadata.fishing_ranges_map[mmsi]
        else:
            ranges = {}

        return np_array_extract_n_random_features(
            random_state, input, num_slices_per_mmsi, max_time_delta,
            window_size, min_timeslice_size, mmsi, ranges)

    (features_list, timestamps, time_bounds_list, mmsis) = tf.py_func(
        replicate_extract, [movement_features, int_mmsi],
        [tf.float32, tf.int32, tf.int32, tf.string])

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
        4. A numpy array with an int64 mmsi for each slice, of dimension [n].

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

    # if slices == []:
    #     # Return an appropriately shaped empty numpy array.
    #     return (np.empty(
    #         [0, 1, window_size, num_features_inc_timestamp - 1],
    #         dtype=np.float32), np.empty(
    #             shape=[0, window_size], dtype=np.int32), np.empty(
    #                 shape=[0, 2], dtype=np.int32), np.empty(
    #                     shape=[0], dtype=np.int64))

    return zip(*slices)


def cropping_all_slice_feature_file_reader(filename_queue, num_features,
                                           time_ranges, window_size,
                                           min_points_for_classification):
    """ Set up a file reader and inference feature extractor for the files in a
        queue.

    An inference feature extractor, pulling all sequential fixed-time slices
    from a vessel movement series.

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
    mmsi = tf.cast(context_features['mmsi'], tf.int64)

    random_state = np.random.RandomState()

    def replicate_extract(input, mmsi):
        return np_array_extract_slices_for_time_ranges(
            random_state, input, num_features, mmsi, time_ranges, window_size,
            min_points_for_classification)

    features_list, timeseries, time_bounds_list, mmsis = tf.py_func(
        replicate_extract, [movement_features, mmsi],
        [tf.float32, tf.int32, tf.int32, tf.int64])

    return features_list, timeseries, time_bounds_list, mmsis


def np_array_extract_all_fixed_slices(input_series, num_features, mmsi,
                                      window_size, shift):
    slices = []
    input_length = len(input_series)
    for end_index in range(input_length, 0, -shift):
        start_index = end_index - window_size
        if start_index < 0:
            continue
        cropped = input_series[start_index:end_index]
        start_time = int(cropped[0][0])
        end_time = int(cropped[-1][0])
        time_bounds = np.array([start_time, end_time], dtype=np.int32)

        assert len(cropped) == window_size
        output_slice = cropped
        without_timestamp = output_slice[:, 1:]
        timeseries = output_slice[:, 0].astype(np.int32)
        slices.append(
            (np.stack([without_timestamp]), timeseries, time_bounds, mmsi))

    if slices == []:
        # Return an appropriately shaped empty numpy array.
        return (np.empty(
            [0, 1, window_size, input_series.shape[1] - 1], dtype=np.float32),
                np.empty(
                    shape=[0, window_size], dtype=np.int32), np.empty(
                        shape=[0, 2], dtype=np.int32), np.empty(
                            shape=[0], dtype=np.int64))

    return zip(*slices)



# TODO: pull out replicate_extract as class and write tests for it.


def process_fixed_window_features(context_features, sequence_features):
    movement_features = sequence_features['movement_features']
    mmsi = tf.cast(context_features['mmsi'], tf.int64)

    if start_date is not None:
        start_stamp = time.mktime(start_date.timetuple())
    if end_date is not None:
        end_stamp = time.mktime(end_date.timetuple())

    def replicate_extract(input_series, mmsi):
        if start_date is not None:
            raw_start_i = np.searchsorted(input_series[:, 0], start_stamp, side='left')
            # If possible go to shift before start so we have good data for whole length
            start_i = max(raw_start_i - shift, 0)
        else:
            start_i = 0
        if end_date is not None:
            raw_end_i = np.searchsorted(input_series[:, 0], end_stamp, side='left')
            # If possible go to shift before end so that we have good data starting at end
            end_i = min(raw_end_i + shift, len(input_series))
        else:
            end_i = len(input_series)
        input_series = input_series[start_i:end_i]
        return np_array_extract_all_fixed_slices(input_series, num_features,
                                                 mmsi, window_size, shift)

    features_list, timeseries, time_bounds_list, mmsis = tf.py_func(
        replicate_extract, [movement_features, mmsi],
        [tf.float32, tf.int32, tf.int32, tf.int64])

    return features_list, timeseries, time_bounds_list, mmsis



def all_fixed_window_feature_file_reader(filename_queue, num_features,
                                         window_size, shift, start_date, end_date):
    """ Set up a file reader and inference feature extractor for the files in a
        queue.

    An inference feature extractor, pulling all sequential fixed-length slices
    from a vessel movement series.

    Args:
        filename_queue: a queue of filenames for feature files to read.
        num_features: the dimensionality of the features.

    Returns:
        A tuple comprising, for the n slices comprising each vessel:
          1. A tensor of the feature slices drawn, of dimension
             [n, 1, window_size, num_features].
          2. A tensor of the timestamps for each feature point of dimension
             [n, window_size].
          3. A tensor of the timebounds for the slices, of dimension [n, 2].
          4. A tensor of the mmsis of each vessel of dimension [n].

    """
    context_features, sequence_features = single_feature_file_reader(
        filename_queue, num_features)

    return process_fixed_window_features(context_features, sequence_features)


def int_or_hash(x):
    try:
        return int(x)
    except:
        return hash(x)

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
        self.mmsi_map_int2str = {int_or_hash(k) : k for k in self.metadata_by_mmsi}


        intersection_mmsis = set(self.metadata_by_mmsi.keys()).intersection(
            set(fishing_ranges_map.keys()))
        logging.info("Metadata for %d mmsis.", len(self.metadata_by_mmsi))
        logging.info("Fishing ranges for %d mmsis.", len(fishing_ranges_map))
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
        assert split in [TRAINING_SPLIT, TEST_SPLIT]
        # Check to make sure we don't have leakage
        assert not (set(self.metadata_by_split[TRAINING_SPLIT].keys()) &
                    set(self.metadata_by_split[TEST_SPLIT].keys())
                    ), 'mmsi in both training and test split'
        return self.metadata_by_split[split].keys()

    def weighted_training_list(self,
                               random_state,
                               split,
                               max_replication_factor,
                               row_filter=lambda row: True,
                               boundary=1):
        replicated_mmsis = []
        logging.info("Training mmsis: %d", len(self.mmsis_for_split(split)))
        fishing_ranges_mmsis = []
        for mmsi, (row, weight) in self.metadata_by_split[split].iteritems():
            if row_filter(row):
                if mmsi in self.fishing_ranges_map:
                    fishing_ranges_mmsis.append(mmsi)
                    weight = weight * self.fishing_range_training_upweight  # TODO: rip this out.

                weight = min(weight, max_replication_factor)

                int_n = int(weight)
                replicated_mmsis += ([mmsi] * int_n)
                frac_n = weight - float(int_n)
                if (random_state.uniform(0.0, 1.0) <= frac_n):
                    replicated_mmsis.append(mmsi)
        missing = (-len(replicated_mmsis)) % boundary
        if missing:
            replicated_mmsis = np.concatenate(
                [replicated_mmsis,
                 np.random.choice(replicated_mmsis, missing)])
        random_state.shuffle(replicated_mmsis)
        logging.info("Replicated training mmsis: %d", len(replicated_mmsis))
        logging.info("Fishing range mmsis: %d", len(fishing_ranges_mmsis))

        return replicated_mmsis

    def fishing_range_only_list(self, random_state, split,
                                max_replication_factor):
        replicated_mmsis = []
        fishing_mmsi_set = set(
            [k for (k, v) in self.fishing_ranges_map.items() if v])
        fishing_range_only_mmsis = [mmsi
                                    for mmsi in self.mmsis_for_split(split)
                                    if mmsi in fishing_mmsi_set]
        logging.info("Fishing range training mmsis: %d / %d",
                     len(fishing_range_only_mmsis),
                     len(self.mmsis_for_split(split)))
        for mmsi in fishing_range_only_mmsis:
            weight = min(self.vessel_weight(mmsi), max_replication_factor)
            assert mmsi in self.fishing_ranges_map
            if weight == 0:
                logging.info('skipping %s due to zero weight', mmsi)
                continue
            int_n = int(weight)
            logging.info(
                "mmis: %s, max_repl_factor: %s, weight: %s, int_n: %s", mmsi,
                max_replication_factor, self.vessel_weight(mmsi), int_n)
            replicated_mmsis += ([mmsi] * int_n)
            frac_n = weight - float(int_n)
            if (random_state.uniform(0.0, 1.0) <= frac_n):
                replicated_mmsis.append(mmsi)

        random_state.shuffle(replicated_mmsis)
        logging.info("Replicated training mmsis: %d", len(replicated_mmsis))

        return replicated_mmsis


def read_vessel_time_weighted_metadata_lines(available_mmsis, lines,
                                             fishing_range_dict):
    """ For a set of vessels, read metadata; use flat weights

    Args:
        available_mmsis: a set of all mmsis for which we have feature data.
        lines: a list of comma-separated vessel metadata lines. Columns are
            the mmsi and a set of vessel type columns, containing at least one
            called 'label' being the primary/coarse type of the vessel e.g.
            (Longliner/Passenger etc.).
        fishing_range_dict: dictionary of mapping mmsi to lists of fishing ranges

    Returns:
        A VesselMetadata object with weights and labels for each vessel.
    """

    metadata_dict = defaultdict(lambda: {})

    # Build a list of vessels + split + and vessel type. Calculate the split on
    # the fly, but deterministically.
    min_time_per_mmsi = np.inf

    for row in lines:
        mmsi = row['mmsi'].strip()
        if mmsi in available_mmsis:
            if mmsi not in fishing_range_dict:
                continue
            # Is this mmsi included only to supress false positives
            # Symptoms; fishing score for this MMSI never different from 0
            is_false_positive = True
            split = row['split']
            if split not in ('Training', 'Test'):
                logging.warning(
                    'MMSI %s has no valid split assigned (%s); using for Training',
                    mmsi, split)
                split = 'Training'
            time_for_this_mmsi = 0
            for rng in fishing_range_dict[mmsi]:
                time_for_this_mmsi += (
                    rng.end_time - rng.start_time).total_seconds()
                if rng.is_fishing > 0:
                    is_false_positive = False
            if time_for_this_mmsi and is_false_positive:
                logging.info('upweighting MMSI %s by %s as a false positive',
                             mmsi, FALSE_POSITIVE_UPWEIGHT)
                time_for_this_mmsi *= FALSE_POSITIVE_UPWEIGHT
            metadata_dict[split][mmsi] = (row, time_for_this_mmsi)
            if time_for_this_mmsi:
                min_time_per_mmsi = min(min_time_per_mmsi, time_for_this_mmsi)

    for split_dict in metadata_dict.values():
        for mmsi in split_dict:
            row, time = split_dict[mmsi]
            split_dict[mmsi] = (row, min(time / min_time_per_mmsi, MAX_UPWEIGHT))

    return VesselMetadata(dict(metadata_dict), fishing_range_dict, 1.0)


def read_vessel_time_weighted_metadata(available_mmsis,
                                       metadata_file,
                                       fishing_range_dict={}):
    reader = metadata_file_reader(metadata_file)

    return read_vessel_time_weighted_metadata_lines(available_mmsis, reader,
                                                    fishing_range_dict)


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
        fishing_range_dict: dictionary of mapping mmsi to lists of fishing ranges
        fishing_range_training_upweight: amount to upweight mmsi with fishing
           ranges to assure adequate coverage.
    Returns:
        A VesselMetadata object with weights and labels for each vessel.
    """

    vessel_type_set = set()
    dataset_kind_counts = defaultdict(lambda: defaultdict(lambda: 0))
    vessel_types = []

    available_mmsis = set(available_mmsis)
    # Build a list of vessels + split + and vessel type. Calculate the split on
    # the fly, but deterministically. Count the occurrence of each vessel type
    # per split.
    for row in lines:
        mmsi = row['mmsi'].strip()
        coarse_vessel_type = row[PRIMARY_VESSEL_CLASS_COLUMN]
        if mmsi in available_mmsis and coarse_vessel_type:
            split = row['split'].strip()
            assert split in ('Training', 'Test'), repr(split)
            vessel_types.append((mmsi, split, coarse_vessel_type, row))
            dataset_kind_counts[split][coarse_vessel_type] += 1
            vessel_type_set.add(coarse_vessel_type)
        # else:
        #     logging.warning('No training data for %s, (%s) %s %s', mmsi, sorted(available_mmsis)[:10], 
        #         type(mmsi), type(sorted(available_mmsis)[0]))

    # Calculate weights for each vessel type per split, for
    # now use weights of sqrt(max_count / count), but eventually weight by prevalance
    # in AIS (as best as we can figure) <== TODO
    dataset_kind_weights = defaultdict(lambda: {})
    for split, counts in dataset_kind_counts.iteritems():
        max_count = max(counts.values())
        for coarse_vessel_type, count in counts.iteritems():
            dataset_kind_weights[split][coarse_vessel_type] = np.sqrt(max_count / float(count))

    metadata_dict = defaultdict(lambda: {})
    for mmsi, split, coarse_vessel_type, row in vessel_types:
        metadata_dict[split][mmsi] = (
            row, dataset_kind_weights[split][coarse_vessel_type])

    if len(vessel_type_set) == 0:
        logging.fatal('No vessel types found for training.')
        sys.exit(-1)

    logging.info("Vessel types: %s", list(vessel_type_set))

    return VesselMetadata(
        dict(metadata_dict), fishing_range_dict,
        fishing_range_training_upweight)


def metadata_file_reader(metadata_file):
    """


    """
    with open(metadata_file, 'r') as f:
        reader = csv.DictReader(f)
        logging.info("Metadata columns: %s", reader.fieldnames)
        for row in reader:
            yield row


def read_vessel_multiclass_metadata(available_mmsis,
                                    metadata_file,
                                    fishing_range_dict={},
                                    fishing_range_training_upweight=1.0):
    reader = metadata_file_reader(metadata_file)

    return read_vessel_multiclass_metadata_lines(
        available_mmsis, reader, fishing_range_dict,
        fishing_range_training_upweight)


def find_available_mmsis(feature_path):
    with tf.Session() as sess:
        logging.info('Reading mmsi list file.')
        root_output_path, _ = os.path.split(feature_path)
        # The feature pipeline stage that outputs the MMSI list is sharded to only
        # produce a single file, so no need to glob or loop here.
        mmsi_list_tensor = tf.read_file(root_output_path +
                                        '/mmsis/part-00000-of-00001.txt')
        els = sess.run(mmsi_list_tensor).split('\n')
        mmsi_list = [mmsi.strip() for mmsi in els if mmsi.strip() != '']

        logging.info('Found %d mmsis.', len(mmsi_list))
        return set(mmsi_list)


def parse_date(date):
    try:
        unix_timestamp = float(date)
        return datetime.datetime.utcfromtimestamp(unix_timestamp).replace(
            tzinfo=pytz.utc)
    except:
        try:
            return dateutil.parser.parse(date)
        except:
            logging.fatal('could not parse date "{}"'.format(data))
            raise


def read_fishing_ranges(fishing_range_file):
    """ Read vessel fishing ranges, return a dict of mmsi to classified fishing
        or non-fishing ranges for that vessel.
    """
    fishing_range_dict = defaultdict(lambda: [])
    with open(fishing_range_file, 'r') as f:
        for l in f.readlines()[1:]:
            els = l.split(',')
            mmsi = els[0].strip()
            start_time = parse_date(els[1])
            end_time = parse_date(els[2])
            is_fishing = float(els[3])

            fishing_range_dict[mmsi].append(
                FishingRange(start_time, end_time, is_fishing))

    return dict(fishing_range_dict)


def build_multihot_lookup_table():
    n_base = len(VESSEL_CLASS_DETAILED_NAMES)
    n_categories = len(VESSEL_CATEGORIES)
    #
    table = np.zeros([n_categories, n_base], dtype=np.int32)
    for i, (_, base_labels) in enumerate(VESSEL_CATEGORIES):
        for lbl in base_labels:
            j = VESSEL_CLASS_DETAILED_NAMES.index(lbl)
            table[i, j] = 1
    return table


multihot_lookup_table = build_multihot_lookup_table()


def multihot_encode(label):
    """Multihot encode based on fine, coarse and is_fishing label

    Args:
        label: Tensor (int)

    Returns:
        Tensor with bits set for every allowable vessel type based on the inputs


    """
    tf_multihot_lookup_table = tf.convert_to_tensor(multihot_lookup_table)
    return tf.gather(tf_multihot_lookup_table, label)
