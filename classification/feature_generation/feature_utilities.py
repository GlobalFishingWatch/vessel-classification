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

import datetime
import pytz
import logging
import numpy as np
import time


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


def empty_data(window_size, series):
    num_features_inc_timestamp = series.shape[1]
    return (np.empty(
        [0, 1, window_size, num_features_inc_timestamp - 1],
        dtype=np.float32), np.empty(
            shape=[0, window_size], dtype=np.int32), np.empty(
                shape=[0, 2], dtype=np.int32), np.empty(
                    shape=[0], dtype=str))     


def cook_features(features, mmsi):
    start_time = int(features[0][0])
    end_time = int(features[-1][0])

    # Drop the first (timestamp) column.
    timestamps = features[:, 0].astype(np.int32)
    features = features[:, 1:]

    if not np.isfinite(features).all():
        logging.fatal('Bad features: %s', features)

    return (np.stack([features]), 
            timestamps, 
            np.array([start_time, end_time], dtype=np.int32), 
            mmsi)


def extract_n_random_fixed_points(random_state, input_series, n,
                                       output_length, mmsi,
                                       selection_ranges):
    """ Extracts a n, random fixed-points slice from a 2d numpy array.
    
    The input array must be 2d, representing a time series, with the first    
    column representing a timestamp (sorted ascending). 

    Args:
        random_state: a numpy randomstate object.
        input_series: the input series. A 2d array first column representing an
            ascending time.  
        n: the number of series to extract
        output_length: the number of points in the output series. Input series    
            shorter than this will be repeated into the output series.   
        mmsi: the mmsi, or vessel_id of the vessel
        selection_ranges: Either a list of time ranges that should be preferentially 
            selected from (we try to get at least on point from one of the ranges), or 
            None if to disable this behaviour. 

    Returns:    
        An array of the same depth as the input, but altered width, representing
        the fixed points slice.   
    """
    input_length = len(input_series)
    num_features_inc_timestamp = input_series.shape[1]
    if input_length < output_length:
        return empty_data(output_length, input_series)

    # Set of points where it would make sense to start a range.
    candidate_set = set()

    starts = [x.start_time_dt for x in selection_ranges]
    starts_ndxs = np.searchsorted(input_series[:, 0], starts, side='left')

    ends = [x.end_time_dt for x in selection_ranges]
    end_ndxs = np.searchsorted(input_series[:, 0], ends, side='right')

    for start_ndx, end_ndx in zip(starts_ndxs, end_ndxs):
        valid_start = max(0, start_ndx - output_length + 1)
        valid_end = min(input_length - output_length + 1, end_ndx)
        candidate_set.update(range(valid_start, valid_end))

    candidates = list(candidate_set)

    if len(candidates) == 0:
        return empty_data(output_length, input_series)

    samples = []
    for _ in range(n):
        start_index = random_state.choice(candidates)
        end_index = start_index + output_length
        samples.append(cook_features(input_series[start_index:end_index], mmsi))

    return zip(*samples)



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

    # TODO: Just figure out the max / min points that are acceptable and pull the random
    # points from there. If none available, return None.
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

    assert max_time_delta != 0

    features = np_array_random_fixed_time_extract(
        random_state, input, max_time_delta, window_size,
        min_timeslice_size)

    if features is None:
        num_features_inc_timestamp = input.shape[1]
        return empty_data(window_size, input)

    return cook_features(features)



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

    if max_time_delta == 0:
        return extract_n_random_fixed_points(random_state, input, n,
                                       window_size, mmsi,
                                       selection_ranges)

    for _ in range(n):
        x = np_array_extract_features(
            random_state, input, max_time_delta, window_size,
            min_timeslice_size, selection_ranges, mmsi)
        samples.append(x)

    return zip(*samples)



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

    return zip(*slices)





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
        slices.append(cook_features(cropped, mmsi))
    if slices == []:
        return empty_data(window_size, input_series)


    return zip(*slices)



