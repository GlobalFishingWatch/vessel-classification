import tensorflow as tf
import logging
import numpy as np
from . import feature_generation
from . import feature_utilities
import six


def input_fn(
             metadata,
             filenames,
             num_features,
             max_time_delta,
             window_size,
             min_timeslice_size,
             objectives,
             parallelism=4,
             num_slices_per_id=4):

    random_state = np.random.RandomState()

    def xform(id_, movement_features):

        def _xform(id_, features):
            id_ = metadata.id_map_int2bytes[id_]
            return feature_utilities.extract_n_random_fixed_times(
                    random_state, features, num_slices_per_id, max_time_delta,
                    window_size, id_, min_timeslice_size)

        features, timestamps, time_ranges, id_ = tf.py_func(
            _xform, 
            [id_, movement_features],
            [tf.float32, tf.int32, tf.int32, tf.string])
        return (features, timestamps, time_ranges, id_)

    def add_labels(features, timestamps, time_bounds, id_):

        def _add_labels(id_, timestamps):
            labels = [o.create_label(id_, timestamps) for o in objectives]
            return labels

        labels =  tf.py_func(
            _add_labels, 
            [id_, timestamps],
            [tf.float32] * len(objectives))
        return ((features, timestamps, time_bounds, id_), tuple(labels))

    def set_shapes(all_features, labels):
        feature_generation.set_feature_shapes(all_features, num_features, window_size)
        for i, obj in enumerate(objectives):
            t = labels[i]
            t.set_shape(obj.output_shape)
        return all_features, labels

    def lbls_as_dict(features, labels):
        d = {obj.name : labels[i] for (i, obj) in enumerate(objectives)}
        return features, d

    def features_as_dict(features, labels):
        features, timestamps, time_bounds, id_ = features
        d = {'features' : features, 'timestamps' : timestamps, 'time_ranges' : time_bounds, 'id' : id_}
        return d, labels

    raw_data = feature_generation.read_input_fn_infinite(
                    filenames,
                    num_features,
                    num_parallel_reads=parallelism,
                    random_state=random_state)

    return (raw_data
                .map(xform, num_parallel_calls=parallelism)
                .flat_map(feature_generation.flatten_features)
                .map(add_labels, num_parallel_calls=parallelism)
                .map(set_shapes, num_parallel_calls=parallelism)
                .map(lbls_as_dict, num_parallel_calls=parallelism)
                .map(features_as_dict, num_parallel_calls=parallelism)
           )


def predict_input_fn(paths,
                   num_features,
                   time_ranges,
                   window_size,
                   min_timeslice_size,
                   parallelism=4):

    random_state = np.random.RandomState()

    def xform(id_, movement_features):

        def _xform(id_, features):
            return feature_utilities.np_array_extract_slices_for_time_ranges(
                    random_state, features, id_, time_ranges,
                    window_size, min_timeslice_size)

        raw_features = tf.cast(movement_features, tf.float32)
        features, timestamps, time_ranges_tensor, id_ = tf.py_func(
            _xform, 
            [id_, raw_features],
            [tf.float32, tf.int32, tf.int32, tf.string])
        return (features, timestamps, time_ranges_tensor, id_)

    def set_shapes(features, timestamps, time_bounds, id_):
        all_features = features, timestamps, time_bounds, id_
        feature_generation.set_feature_shapes(all_features, num_features, window_size)
        return all_features

    def features_as_dict(features, timestamps, time_bounds, id_):
        d = {'features' : features, 'timestamps' : timestamps, 'time_ranges' : time_bounds, 'id' : id_}
        return d

    raw_data = feature_generation.read_input_fn_one_shot(paths, num_features, num_parallel_reads=parallelism)

    return (raw_data
                .map(xform, num_parallel_calls=parallelism)
                .flat_map(feature_generation.flatten_features)
                .map(set_shapes)
                .map(features_as_dict)
           )