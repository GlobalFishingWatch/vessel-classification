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
import six
import time


EPOCH_DT = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)


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
    if reps > 1:
        return np.concatenate([slice] * reps, axis=0)[:window_size]
    else:
        return slice[:window_size].copy()

def np_zero_pad_slice(slice, window_size, random_state):
    """ Pads slice to the specified window size.

  Series that are shorter than window_size are repeated into unfilled space.

  Args:
    slice: np.array.
    window_size: int
        Size the array must be padded to.
    random_state: np.RandomState

  Returns:
    a numpy array of length window_size in the first dimension.

    TODO: this function has an inaccurate name, really this is doing
    pad_repeat_slice with a random offset for data augmentation. 
    Rename or remove at next cleanup since it doesn't appear to be
    used..
  """

    slice_length = len(slice)
    delta = window_size - slice_length
    assert delta >= 0
    offset = random_state.randint(0, delta + 1)
    return np.concatenate([slice] * reps, axis=0)[offset:offset+window_size]

def np_pad_repeat_slice_2(slice, window_size, random_state):
    """ Pads slice to the specified window size then rolls them.

  Series that are shorter than window_size are repeated into unfilled space,
  then the series is randomly rolled along the time axis to generate more 
  training diversity. This has the side effect of adding a non-physical 
  seam in the data, but in practice seems to work better than not rolling.

  Similar to `np_pad_repeat_slice` except for rolling the sequence along the
  time axis.

  Args:
    slice: a numpy array.
    window_size: the size the array must be padded to.
    random_state: a numpy RandomState object.

  Returns:
    a numpy array of length window_size in the first dimension.
  """

    slice_length = len(slice)
    delta = window_size - slice_length
    assert delta >= 0
    slice = slice.copy()
    GAP_LOGDT = 100
    slice[0, 1] = GAP_LOGDT
    reps = int(np.ceil(window_size / float(slice_length)))
    repeated = np.concatenate([slice] * reps, axis=0)
    offset = random_state.randint(0, window_size)
    return np.roll(repeated, offset, axis=0)[:window_size]

def np_array_random_fixed_length_extract(random_state, input_series,
                                         output_length):

    cropped = input_series[start_offset:end_offset]

    return np_pad_repeat_slice_2(cropped, output_length, random_state)


def empty_data(window_size, series):
    num_features_inc_timestamp = series.shape[1]
    return (np.empty(
        [0, 1, window_size, num_features_inc_timestamp - 1],
        dtype=np.float32), np.empty(
            shape=[0, window_size], dtype=np.int32), np.empty(
                shape=[0, 2], dtype=np.int32), np.empty(
                    shape=[0], dtype=str))     


def cook_features(features, id_):
    """Convert raw features into something the model can digest

        Args:
            features: np.array of raw features
            id_ : the id associated with these features.

        Returns:
            (2D np.array of features for the model,
             2D np.array of timestamps,
             (start_time, end_time) as seconds from Unix epoch.
             str id of vessel)
    """
    # We use min and max here to account for possible rolling / replicating
    # that goes on elsewhere.
    start_time = int(features[:, 0].min())
    end_time = int(features[:, 0].max())

    # Drop the first (timestamp) column.
    timestamps = features[:, 0].astype(np.int32)
    features = features[:, 1:]

    if not np.isfinite(features).all():
        logging.fatal('Bad features: %s', features)

    return (np.stack([features]), 
            timestamps, 
            np.array([start_time, end_time], dtype=np.int32), 
            id_)


def extract_n_random_fixed_points(random_state, input_series, n,
                                       output_length, id_,
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
        id_: the id of the vessel
        selection_ranges: Either a list of time ranges that should be preferentially 
            selected from (we try to get at least on point from one of the ranges), or 
            None if to disable this behaviour. 

    Returns:    
        An array of the same depth as the input, but altered width, representing
        the fixed points slice.   
    """
    input_length = len(input_series)
    if input_length < output_length:
        return empty_data(output_length, input_series)

    # Set of points where it would make sense to start a range.
    candidate_set = set()

    starts = [(x.start_time - EPOCH_DT).total_seconds() for x in selection_ranges]
    starts_ndxs = np.searchsorted(input_series[:, 0], starts, side='left')

    ends = [(x.end_time - EPOCH_DT).total_seconds() for x in selection_ranges]
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
        samples.append(cook_features(input_series[start_index:end_index], id_))

    return list(zip(*samples))




def setup_cook_features_into(n, features_shape):
    s0, s1 = features_shape
    features = np.empty((n, s0, s1 - 1), np.float32)
    timestamps = np.empty([n, s0], np.int32)
    ranges = np.empty([n, 2], np.int32)
    ids = np.empty([n], object)
    return features, timestamps, ranges, ids

def cook_features_into(arrays, i, data, id_, range_=None):
    """Convert raw features into something the model can digest

        Args:
            arrays: (features, timestamps, ranges)
            i : index to inserrt into
            features: np.array of raw features
            range: tuple, optional
                (start, stop) timestamps, if not supplied, this is derived
                from features.

        Returns:
            (2D np.array of features for the model,
             2D np.array of timestamps,
             (start_time, end_time) as seconds from Unix epoch.
             str id of vessel)
    """
    features, timestamps, ranges, ids = arrays
    features[i] = data[:, 1:]
    timestamps[i] = data[:, 0]
    if range_ is None:
        ranges[i, 0] = int(data[:, 0].min())
        ranges[i, 1] = int(data[:, 0].max())
    else:
        ranges[i, :] = range_
    ids[i] = id_

def extract_n_random_fixed_times(random_state, input_series, n,
                                       max_time_delta, output_length,
                                       id_, min_timeslice_size):
    """ Extracts a random fixed-time slice from a 2d numpy array.
    
    The input array must be 2d, representing a time series, with the first    
    column representing a timestamp (sorted ascending). Any values in the series    
    with a time greater than (first time + max_time_delta) are removed and the    
    prefix series repeated into the window to pad. 
    Args:
        random_state: a numpy randomstate object.
        input_series: the input series. A 2d array first column representing an
            ascending time.   
        n: the number of series to extract
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
    input_length = len(input_series)
    if input_length < min_timeslice_size:
        return empty_data(output_length, input_series)

    min_time = input_series[0, 0] - (output_length - min_timeslice_size)
    max_ndx = input_length - min_timeslice_size
    max_time_due_to_ndx = input_series[max_ndx, 0]
    max_time_due_to_time = input_series[-1, 0] - max_time_delta
    max_time = min(max_time_due_to_ndx, max_time_due_to_time)

    if max_time < min_time:
        return empty_data(output_length, input_series)

    # TODO: clarify by breaking into two function
    arrays = setup_cook_features_into(n, (output_length, input_series.shape[-1]))
    for i in range(n):
        start_time = random_state.randint(min_time, max_time + 1)
        start_index = np.searchsorted(input_series[:, 0], start_time, side='left')
        end_index = start_index + output_length
        cropped = input_series[start_index:end_index] # Might only have min_timeslice_size points
        padded = np_pad_repeat_slice(cropped, output_length)
        cook_features_into(arrays, i, padded, id_)

    return arrays




def np_array_extract_slices_for_time_ranges(
        random_state, input_series, id_,
        time_ranges, window_size, min_points_for_classification):
    """ Extract and process a set of specified time slices from a vessel
        movement feature.

    Args:
        random_state: a numpy randomstate object.
        input: the input data as a 2d numpy array.
        id_: the id of the vessel which made this series.
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
        4. A numpy array with an int64 id for each slice, of dimension [n].

    """
    times = input_series[:, 0]
    ndx = 0
    arrays = setup_cook_features_into(len(time_ranges), 
                        (window_size, input_series.shape[-1]))
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
            padded = np_pad_repeat_slice(cropped, window_size)
            cook_features_into(arrays, ndx, padded, id_, (start_time, end_time))
            ndx += 1
    return tuple(x[:ndx] for x in arrays)

def np_array_extract_all_fixed_slices(input_series, num_features, id_,
                                      window_size, shift):
    slices = []
    input_length = len(input_series)
    for end_index in range(input_length, 0, -shift):
        start_index = end_index - window_size
        if start_index < 0:
            if start_index != -shift:
                logging.warning('input not correctly padded, dropping start')
            continue
        cropped = input_series[start_index:end_index]
        start_time = int(cropped[0][0])
        end_time = int(cropped[-1][0])
        time_bounds = np.array([start_time, end_time], dtype=np.int32)
        assert len(cropped) == window_size
        slices.append(cook_features(cropped, id_))
    if slices == []:
        return empty_data(window_size, input_series)
    return list(zip(*slices))



def process_fixed_window_features(random_state, features, id_, 
        num_features, window_size, shift, start_date, end_date, 
        win_start, win_end):
    """Extract sequential sets of features with a given window size.

    This will return multiple sets of training features. Each set
    will be `window_size` in length and each successive set will be
    shifted by `shift`.  The algorithm will attempt to place the point
    `win_start` within the first window at `start_data` and `win_end`
    within the last window at `end_data`.

    Note that `shift` must equal `win_end `- `win_start`.

    Args:
        random_state: a numpy randomstate object.
        features: the input data as a 2d numpy array.
        id_: the id of the vessel which made this series.
        num_features: the number of features in the raw features.
        window_size: the size of the window in points.
        shift: how many points the window should shift between feature sets.
        start_date: Attempt to place first point of first window here.
        end_data: Attempt to place last point of last window here.
        win_start: first point within in the window where we extact data.
        win_end: last point within the window where we extract data.

    """
    assert win_end - win_start == shift + 1, (win_end, win_start, shift)
    pad_end = window_size - win_end
    pad_start = win_start

    if len(features) == 0:
        return empty_data(window_size, features)

    is_sorted = all(features[i, 0] <= features[i+1, 0] 
                    for i in six.moves.range(len(features)-1))
    assert is_sorted

    if start_date is not None:
        start_stamp = time.mktime(start_date.timetuple())
    if end_date is not None:
        end_stamp = time.mktime(end_date.timetuple())

    if end_date is not None:
        raw_end_i = np.searchsorted(features[:, 0], end_stamp, side='right')
        # How much padding do we need:
        available = len(features) - raw_end_i
        n_pad_end = max(pad_end - available, 0)
    else:
        raw_end_i = len(features)
        n_pad_end = pad_end
    end_i = raw_end_i + pad_end

    #    \/ raw_start_i
    #              ppBBBBpp<--- shift
    # ppBBBBpp.........ppBBBBpp
    #   end_i - winsize^       ^ end_i

    if start_date is not None:
        # If possible go to shift before pad so we have good data for whole length
        raw_start_i = np.searchsorted(features[:, 0], start_stamp, side='left')
    else:
        raw_start_i = 0
    assert raw_start_i >= 0

    start_i = raw_start_i - pad_start

    if start_i >= len(features) or end_i < 1:
        # No features overlap with window
        return empty_data(window_size, features)

    while (end_i - start_i - window_size) % shift:
        start_i -= 1

    if n_pad_end > 0:
        features = np.concatenate([features, n_pad_end * [features[-1]]], axis=0)
    else:
        features = features[:end_i]
    assert len(features) == end_i

    if start_i < 0:
        features = np.concatenate([features, (-start_i) * [features[0]]], axis=0)
    else:
        features = features[start_i:]
    assert len(features) == end_i - start_i

    assert (end_i - start_i - window_size) % shift == 0

    return np_array_extract_all_fixed_slices(features, num_features,
                                             id_, window_size, shift)



