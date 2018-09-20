import tensorflow as tf
import numpy as np
from . import utility



def input_fn(vessel_metadata,
             filenames,
            num_features,
            max_time_delta,
            window_size,
            min_timeslice_size,
            add_labels_fn,
            set_labels_shape_fn,
            num_parallel_reads=4):
    
    def parse_function(example_proto):
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
    num_slices_per_mmsi = 8

    def _xform(features, int_mmsi):
        # Extract several random windows from each vessel track
        # TODO: Fix feature generation so it returns strings directly
        mmsi = vessel_metadata.mmsi_map_int2str[int_mmsi]
        ranges = vessel_metadata.fishing_ranges_map.get(mmsi, {})
        return utility.np_array_extract_n_random_features(
                random_state, features, num_slices_per_mmsi, max_time_delta,
                window_size, min_timeslice_size, mmsi, ranges)
    
    def xform(mmsi, movement_features):
        int_mmsi = tf.cast(mmsi, tf.int64)
        features = tf.cast(movement_features, tf.float32)
        features, timestamps, time_ranges, mmsi = tf.py_func(
            _xform, 
            [features, int_mmsi],
            [tf.float32, tf.int32, tf.int32, tf.string])
        features = tf.squeeze(features, axis=1)
        return tf.data.Dataset.from_tensor_slices((features, timestamps, time_ranges, mmsi))

    def set_shapes(all_features, labels):
        features, timestamps, time_ranges, mmsi = all_features
        features.set_shape([window_size, num_features - 1]) 
        timestamps.set_shape([window_size])
        time_ranges.set_shape([2])
        mmsi.set_shape([])
        labels = set_labels_shape_fn(labels)
        return ((features, timestamps, time_ranges, mmsi), labels)

    path_ds = (tf.data.Dataset.from_tensor_slices(filenames)
                    .repeat()
                    .shuffle(len(filenames))
                    )

    return (tf.data.TFRecordDataset(path_ds, num_parallel_reads=num_parallel_reads)
                .map(parse_function)
                .flat_map(xform) # This makes multiple small slices from each file
                .map(add_labels_fn)
                .map(set_shapes)
           )

