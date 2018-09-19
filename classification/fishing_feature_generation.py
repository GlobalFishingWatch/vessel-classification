import numpy as np
import tensorflow as tf
import calendar
from . import utility


def input_fn(vessel_metadata,
             filenames,
            num_features,
            max_time_delta,
            window_size,
            min_timeslice_size,
            select_ranges=False,
            num_parallel_reads=4):
    
    def _parse_function(example_proto):
        context_features, sequence_features = tf.parse_single_sequence_example(
            example_proto,
            context_features={
                'mmsi': tf.FixedLenFeature([], tf.int64)
                },
            sequence_features={
                'movement_features': tf.FixedLenSequenceFeature(shape=(num_features, ), 
                                                                dtype=tf.float32)
            }
        )
        return context_features['mmsi'], sequence_features['movement_features']
    
    random_state = np.random.RandomState()
    vmd_mmsi_map = vessel_metadata.mmsi_map_int2str
    vmd_fr_map = vessel_metadata.fishing_ranges_map
    num_slices_per_mmsi = 8

    def replicate_extract(features, int_mmsi):
        # Extract several random windows from each vessel track
        # TODO: Fix feature generation so it returns strings directly
        mmsi = vmd_mmsi_map[int_mmsi]
        
        if mmsi in vessel_metadata.fishing_ranges_map:
            ranges = vmd_fr_map[mmsi]
        else:
            ranges = {}

        return utility.np_array_extract_n_random_features(
            random_state, features, num_slices_per_mmsi, max_time_delta,
            window_size, min_timeslice_size, mmsi, ranges)
    
        
    def xform(mmsi, movement_features):
        int_mmsi = tf.cast(mmsi, tf.int64)
        features = tf.cast(movement_features, tf.float32)
        features, timestamps, time_ranges, mmsi = tf.py_func(
            replicate_extract, 
            [features, int_mmsi],
            [tf.float32, tf.int32, tf.int32, tf.string])
        features = tf.squeeze(features, axis=1)
        return tf.data.Dataset.from_tensor_slices((features, timestamps, time_ranges, mmsi))


    def _add_labels(mmsi, timestamps):
        dense_labels = np.zeros_like(timestamps, dtype=np.float32)
        dense_labels.fill(-1.0)
        if mmsi in vessel_metadata.fishing_ranges_map:
            for (start_time, end_time, is_fishing
                 ) in vessel_metadata.fishing_ranges_map[mmsi]:
                start_range = calendar.timegm(start_time.utctimetuple(
                ))
                end_range = calendar.timegm(end_time.utctimetuple())
                mask = (timestamps >= start_range) & (
                    timestamps <= end_range)
                dense_labels[mask] = is_fishing
        return dense_labels


    def add_labels(features, timestamps, time_bounds, mmsi):
        [labels] =  tf.py_func(
            _add_labels, 
            [mmsi, timestamps],
            [tf.float32])
        return ((features, timestamps, time_bounds, mmsi), labels)

    def set_shapes(all_features, labels):
        features, timestamps, time_ranges, mmsi = all_features
        features.set_shape([window_size, num_features - 1]) 
        timestamps.set_shape([window_size])
        time_ranges.set_shape([2])
        mmsi.set_shape([])
        labels.set_shape([window_size])
        return ((features, timestamps, time_ranges, mmsi), labels)


    path_ds = (tf.data.Dataset.from_tensor_slices(filenames)
                    .shuffle(len(filenames))
                    .repeat())

    return (tf.data.TFRecordDataset(path_ds, num_parallel_reads=num_parallel_reads)
                .map(_parse_function)
                .flat_map(xform) # This makes multiple small slices from each file
                .map(add_labels)
                .map(set_shapes)
           )



# STUFF Below is for experimental adding fishing information to fishing

ENCODED_LABELS = [
    'drifting_longlines',
    'trawlers'
]

for x in ENCODED_LABELS:
    assert x in utility.VESSEL_CLASS_DETAILED_NAMES

ONE_HOT_COUNT = len(ENCODED_LABELS) + 1

encoding = {x : i + 1 for (i, x) in enumerate(ENCODED_LABELS)}

def encode(label):
    encoding.get(label, 0)




def np_array_extract_n_random_features(random_state, input, n, max_time_delta,
                                       window_size, min_timeslice_size, mmsi,
                                       selection_ranges, encoded_label):
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
        features, timestamps, time_bounds = utility.np_array_extract_features(
            random_state, input, max_time_delta, window_size,
            min_timeslice_size, selection_ranges, mmsi)

        one_hot_labels = np.zeros([len(features), ONE_HOT_COUNT], dtype=features.dtype)
        one_hot_labels[:, encoded_label] = 1
        features = np.concatenate([features, one_hot_labels], axis=-1)

        samples.append((np.stack([features]), timestamps, time_bounds, mmsi))

    return zip(*samples)

def training_features(vessel_metadata,
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
    context_features, sequence_features = utility.single_feature_file_reader(
        filename_queue, num_features - ONE_HOT_COUNT)

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

        label = vessel_metadata.vessel_label(utility.PRIMARY_VESSEL_CLASS_COLUMN, mmsi)

        return np_array_extract_n_random_features(
            random_state, input, num_slices_per_mmsi, max_time_delta,
            window_size, min_timeslice_size, mmsi, ranges, encode(label))

    (features_list, timestamps, time_bounds_list, mmsis) = tf.py_func(
        replicate_extract, [movement_features, int_mmsi],
        [tf.float32, tf.int32, tf.int32, tf.string])

    return features_list, timestamps, time_bounds_list, mmsis