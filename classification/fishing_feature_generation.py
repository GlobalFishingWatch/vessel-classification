import calendar
import numpy as np
import tensorflow as tf
import feature_generation


def input_fn(vessel_metadata,
             filenames,
            num_features,
            max_time_delta,
            window_size,
            min_timeslice_size,
            num_parallel_reads=4):

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

    def set_labels_shape(labels):
        labels.set_shape([window_size])
        return labels

    return feature_generation.input_fn(
                vessel_metadata,
                filenames,
                num_features,
                max_time_delta,
                window_size,
                min_timeslice_size,
                num_parallel_reads=num_parallel_reads,
                add_labels_fn=add_labels,
                set_labels_shape_fn=set_labels_shape
        )