import datetime
import logging
import numpy as np
import tempfile
import subprocess
import os
import resource
import shutil
import time

import tensorflow as tf

from .utility import np_array_extract_all_fixed_slices
from .utility import np_array_extract_slices_for_time_ranges
from .utility import np_pad_repeat_slice


class GCSFile(object):

    def __init__(self, path):
        self.gcs_path = path

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(self.temp_dir, os.path.basename(self.gcs_path))
        subprocess.check_call(['gsutil', 'cp', self.gcs_path, local_path])
        return self._process(local_path)

    def _process(self, path):
        return open(path, 'rb')

    def __exit__(self, *args):
        shutil.rmtree(self.temp_dir)

class GCSExampleIter(object):

    def __init__(self, path):
        self.gcs_path = path

    def __enter__(self):
        return self._process(self.gcs_path)

    def _process(self, path):
        return tf.python_io.tf_record_iterator(path)

    def __exit__(self, *args):
        pass


# If we keep building deserializers it leaks memory, so build one once and keep it around.
class Deserializer(object):

    def __init__(self, num_features, sess=None):
        self.num_features = num_features
        self.serialized_example = tf.placeholder(tf.string, shape=())
        context_features, sequence_features = tf.parse_single_sequence_example(
            self.serialized_example,
            # Defaults are not specified since both keys are required.
            context_features={'mmsi': tf.FixedLenFeature([], tf.int64), },
            sequence_features={
                'movement_features': tf.FixedLenSequenceFeature(
                    shape=(num_features, ), dtype=tf.float32)
        }) 
        self.context_features = context_features
        self.sequence_features = sequence_features
        self.sess = sess

    def __call__(self, serialized_example):
        if self.sess is None:
            sess = tf.get_default_session()
        else:
            sess = self.sess
        return sess.run([self.context_features, self.sequence_features], 
                feed_dict={self.serialized_example: serialized_example})




# Can we unify this with the version in utility?

def process_fixed_window_features(context_features, sequence_features, 
        num_features, window_size, shift, start_date, end_date, win_start, win_end):
    
    features = sequence_features['movement_features']
    mmsi = context_features['mmsi']

    assert win_end - win_start == shift, (win_end, win_start, shift)
    pad_end = window_size - win_end
    pad_start = win_start

    is_sorted = all(features[i, 0] <= features[i+1, 0] for i in xrange(len(features)-1))
    assert is_sorted

    if start_date is not None:
        start_stamp = time.mktime(start_date.timetuple())
    if end_date is not None:
        end_stamp = time.mktime(end_date.timetuple())

    if end_date is not None:
        raw_end_i = np.searchsorted(features[:, 0], end_stamp, side='right')
        # If possible go to pad after end so that we have good data starting at end
        end_i = min(raw_end_i + pad_end, len(features))
    else:
        end_i = len(features)

    #    \/ raw_start_i
    #              ppBBBBpp<--- shift
    # ppBBBBpp.........ppBBBBpp
    #   end_i - winsize^       ^ end_i

    if start_date is not None:
        # If possible go to shift before pad so we have good data for whole length
        raw_start_i = np.searchsorted(features[:, 0], start_stamp, side='left') - pad_start
    else:
        raw_start_i = 0

    if end_i < window_size:
        # There aren't enough points to classify, so pad by replicating the first point.
        # Do this here so raw_start_i is calculated on actual features
        count = window_size - end_i
        extra = [features[0]] * count
        features = np.concatenate([extra, features], axis=0)
        end_i += count
        raw_start_i += count

    # Now clean up raw_start. 
    #   First add enough points that we are at the beginning of a shift.
    delta = ((end_i - window_size) - raw_start_i) % shift

    if delta == 0:
        start_i = raw_start_i
    else:
        start_i = raw_start_i + delta - shift

    # Shift backwards till we can hold at least one window
    while end_i - start_i < window_size:
        start_i -= shift

    # Now shift forward till we are nonnegative:
    while start_i < 0:
        start_i += shift

    features = features[start_i:end_i]

    return np_array_extract_all_fixed_slices(features, num_features,
                                             mmsi, window_size, shift)




def all_fixed_window_feature_file_iterator(filenames, deserializer,
                                         window_size, shift, start_date, end_date,
                                         win_start, win_end):
    """ Set up a file reader and inference feature extractor for the specified files

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
    for path in filenames:
        with GCSExampleIter(path) as exmpliter:
            for exmp in exmpliter:
                context_features, sequence_features = deserializer(exmp) 
                for values in zip(*process_fixed_window_features(context_features, 
                                        sequence_features, deserializer.num_features, 
                                        window_size, shift, start_date, end_date, win_start, win_end)):
                    yield values



def process_all_slice_features(context_features, sequence_features, 
        time_ranges, window_size, min_points_for_classification, num_features):

    movement_features = sequence_features['movement_features']
    mmsi = context_features['mmsi']

    random_state = np.random.RandomState()

    def replicate_extract(input, mmsi):
        return np_array_extract_slices_for_time_ranges(
            random_state, input, num_features, mmsi, time_ranges, window_size,
            min_points_for_classification)

    return replicate_extract(movement_features, mmsi)



def cropping_all_slice_feature_file_iterator(filenames, deserializer,
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
    for path in filenames:
        with GCSExampleIter(path) as exmpliter:
            for exmp in exmpliter:
                context_features, sequence_features = deserializer(exmp) 
                for values in zip(*process_all_slice_features(
                        context_features, sequence_features, time_ranges, 
                        window_size, min_points_for_classification, deserializer.num_features)):
                    yield values


