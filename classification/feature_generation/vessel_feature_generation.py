import tensorflow as tf
import feature_generation
import numpy as np
from classification import metadata
from . import feature_generation
from . import feature_utilities


def input_fn(vessel_metadata,
             filenames,
            num_features,
            max_time_delta,
            window_size,
            min_timeslice_size,
            objectives,
            parallelism=4,
            num_slices_per_mmsi=8):

    random_state = np.random.RandomState()

    def xform(mmsi, movement_features):

        def _xform(features, int_mmsi):
            mmsi = vessel_metadata.mmsi_map_int2str[int_mmsi]
            return feature_utilities.extract_n_random_fixed_times(
                    random_state, features, num_slices_per_mmsi, max_time_delta,
                    window_size, mmsi, min_timeslice_size)

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
            labels = [o.create_label(mmsi, timestamps) for o in objectives]
            return labels

        labels =  tf.py_func(
            _add_labels, 
            [mmsi, timestamps],
            [tf.float32] * len(objectives))
        return ((features, timestamps, time_bounds, mmsi), tuple(labels))

    def set_shapes(all_features, labels):
        class_count = len(metadata.VESSEL_CLASS_DETAILED_NAMES)
        feature_generation.set_feature_shapes(all_features, num_features, window_size)
        for i, obj in enumerate(objectives):
            t = labels[i]
            t.set_shape(obj.output_shape)
        return all_features, labels

    def lbls_as_dict(features, labels):
        d = {obj.name : labels[i] for (i, obj) in enumerate(objectives)}
        return features, d

    raw_data = feature_generation.read_input_fn(
                    vessel_metadata,
                    filenames,
                    num_features,
                    num_parallel_reads=parallelism,
                    random_state=random_state)

    return (raw_data
                .map(xform, num_parallel_calls=parallelism)
                .flat_map(feature_generation.flatten_features)
                .map(add_labels, num_parallel_calls=parallelism)
                .map(set_shapes)
                .map(lbls_as_dict)
           )

