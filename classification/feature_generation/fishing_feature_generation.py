import calendar
import numpy as np
import tensorflow as tf
from . import feature_generation
from . import feature_utilities

def input_fn(vessel_metadata,
             filenames,
            num_features,
            max_time_delta,
            window_size,
            min_timeslice_size,
            parallelism=4,
            num_slices_per_mmsi=8):

    random_state = np.random.RandomState()

    
    def xform(mmsi, movement_features):

        def _xform(features, int_mmsi):
            # Extract several random windows from each vessel track
            # TODO: Fix feature generation so it returns strings directly
            mmsi = vessel_metadata.mmsi_map_int2str[int_mmsi]
            ranges = vessel_metadata.fishing_ranges_map.get(mmsi, {})
            return feature_utilities.extract_n_random_fixed_points(
                            random_state, features, num_slices_per_mmsi,
                            window_size, mmsi, ranges)

        int_mmsi = tf.cast(mmsi, tf.int64)
        features = tf.cast(movement_features, tf.float32)
        features, timestamps, time_ranges, mmsi = tf.py_func(
            _xform, 
            [features, int_mmsi],
            [tf.float32, tf.int32, tf.int32, tf.string])
        features = tf.squeeze(features, axis=1)
        return (features, timestamps, time_ranges, mmsi)


    def add_labels(features, timestamps, time_bounds, mmsi):

        def _add_labels(mmsi, timestamps):
            dense_labels = np.zeros_like(timestamps, dtype=np.float32)
            dense_labels.fill(-1.0)
            if mmsi in vessel_metadata.fishing_ranges_map:
                for sel_range in vessel_metadata.fishing_ranges_map[mmsi]:
                    start_range = calendar.timegm(sel_range.start_time.utctimetuple(
                    ))
                    end_range = calendar.timegm(sel_range.end_time.utctimetuple())
                    mask = (timestamps >= start_range) & (
                        timestamps <= end_range)
                    dense_labels[mask] = sel_range.is_fishing
            return dense_labels

        [labels] =  tf.py_func(
            _add_labels, 
            [mmsi, timestamps],
            [tf.float32])
        return ((features, timestamps, time_bounds, mmsi), labels)

    def set_shapes(all_features, labels):
        feature_generation.set_feature_shapes(all_features, num_features, window_size)
        labels.set_shape([window_size])
        return all_features, labels

    raw_data = feature_generation.read_input_fn_infinite(
                    filenames,
                    num_features,
                    num_parallel_reads=parallelism,
                    random_state=random_state)

    return (raw_data
                .map(xform, num_parallel_calls=parallelism)
                .flat_map(feature_generation.flatten_features)
                .map(add_labels, num_parallel_calls=parallelism)
                .map(set_shapes)
           )


