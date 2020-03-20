import calendar
import numpy as np
import os
import tensorflow as tf
import six
from . import feature_generation
from . import feature_utilities

def input_fn(metadata,
            filenames,
            num_features,
            max_time_delta,
            window_size,
            min_timeslice_size,
            parallelism=4,
            num_slices_per_id=8,
            num_parallel_reads=1):

    random_state = np.random.RandomState()

    weights = []
    for p in filenames:
        id_, _ = os.path.splitext(os.path.basename(p))
        id_ = six.ensure_binary(id_)
        weights.append(metadata.vessel_weight(id_))
    
    def xform(id_, movement_features):

        def _xform(id_, features):
            # Extract several random windows from each vessel track
            id_ = metadata.id_map_int2bytes[id_]
            ranges = metadata.fishing_ranges_map.get(id_, {})
            return feature_utilities.extract_n_random_fixed_points(
                            random_state, features, num_slices_per_id,
                            window_size, id_, ranges)

        features, timestamps, time_ranges, id_ = tf.compat.v1.py_func(
            _xform, 
            [id_, movement_features],
            [tf.float32, tf.int32, tf.int32, tf.string])
        features = tf.squeeze(features, axis=1)
        return (features, timestamps, time_ranges, id_)

    fishing_ranges_map = {}
    for k, v in metadata.fishing_ranges_map.items():
        fishing_ranges_map[k] = []
        for sel_range in v:
            start_range = calendar.timegm(sel_range.start_time.utctimetuple())
            end_range = calendar.timegm(sel_range.end_time.utctimetuple())
            fishing_ranges_map[k].append((start_range, end_range, sel_range.is_fishing))

    def add_labels(features, timestamps, time_bounds, id_):

        def _add_labels(id_, timestamps):
            dense_labels = np.empty_like(timestamps, dtype=np.float32)
            dense_labels.fill(-1.0)
            if id_ in fishing_ranges_map:
                for start_range, end_range, is_fishing in fishing_ranges_map[id_]:
                    start_ndx = np.searchsorted(timestamps, start_range, side='left')
                    end_ndx = np.searchsorted(timestamps, end_range, side='right')
                    dense_labels[start_ndx:end_ndx] = is_fishing
            return dense_labels

        [labels] =  tf.compat.v1.py_func(
            _add_labels, 
            [id_, timestamps],
            [tf.float32])
        return ((features, timestamps, time_bounds, id_), labels)

    def set_shapes(all_features, labels):
        feature_generation.set_feature_shapes(all_features, num_features, window_size)
        labels.set_shape([window_size])
        return all_features, labels

    def features_as_dict(features, labels):
        features, timestamps, time_bounds, id_ = features
        d = {'features' : features, 'timestamps' : timestamps, 'time_ranges' : time_bounds, 'id' : id_}
        return d, labels

    raw_data = feature_generation.read_input_fn_infinite(
                    filenames,
                    num_features,
                    num_parallel_reads=num_parallel_reads,
                    random_state=random_state,
                    weights=weights)

    return (raw_data
                .map(xform, num_parallel_calls=parallelism)
                .flat_map(feature_generation.flatten_features)
                .map(add_labels, num_parallel_calls=parallelism)
                .map(set_shapes)
                .map(features_as_dict)
           )




def predict_input_fn(paths,
                   num_features,
                   window_size,
                   start_date,
                   end_date,
                   window,
                   parallelism=4):

    if window is None:
        b, e = 0, window_size
    else:
        b, e = window
    shift = e - b - 1

    random_state = np.random.RandomState()



    # TODO: use paths to build hashlist and test
    # Look again at differences between fishing and vessel inference



    def xform(id_, movement_features):

        def _xform(id_, features):
            return feature_utilities.process_fixed_window_features(
                    random_state, features, id_, num_features, 
                    window_size, shift, start_date, end_date, b, e)

        features, timestamps, time_ranges_tensor, id_ = tf.compat.v1.py_func(
            _xform, 
            [id_, movement_features],
            [tf.float32, tf.int32, tf.int32, tf.string])
        features = tf.squeeze(features, axis=1)
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


