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
            num_slices_per_id=8):

    random_state = np.random.RandomState()

    
    def xform(id_, movement_features):

        def _xform(features, int_id):
            # Extract several random windows from each vessel track
            # TODO: Fix feature generation so it returns strings directly
            id_ = vessel_metadata.id_map_int2str[int_id]
            ranges = vessel_metadata.fishing_ranges_map.get(id_, {})
            return feature_utilities.extract_n_random_fixed_points(
                            random_state, features, num_slices_per_id,
                            window_size, id_, ranges)

        int_id = tf.cast(id_, tf.int64)
        features = tf.cast(movement_features, tf.float32)
        features, timestamps, time_ranges, id_ = tf.py_func(
            _xform, 
            [features, int_id],
            [tf.float32, tf.int32, tf.int32, tf.string])
        features = tf.squeeze(features, axis=1)
        return (features, timestamps, time_ranges, id_)


    def add_labels(features, timestamps, time_bounds, id_):

        def _add_labels(id_, timestamps):
            dense_labels = np.zeros_like(timestamps, dtype=np.float32)
            dense_labels.fill(-1.0)
            if id_ in vessel_metadata.fishing_ranges_map:
                for sel_range in vessel_metadata.fishing_ranges_map[id_]:
                    start_range = calendar.timegm(sel_range.start_time.utctimetuple(
                    ))
                    end_range = calendar.timegm(sel_range.end_time.utctimetuple())
                    mask = (timestamps >= start_range) & (
                        timestamps <= end_range)
                    dense_labels[mask] = sel_range.is_fishing
            return dense_labels

        [labels] =  tf.py_func(
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
                    num_parallel_reads=parallelism,
                    random_state=random_state)

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
    shift = e - b

    random_state = np.random.RandomState()

    def xform(id_, movement_features):

        def _xform(features, int_id):
            id_ = str(int_id)
            return feature_utilities.process_fixed_window_features(
                    random_state, features, id_, num_features, 
                    window_size, shift, start_date, end_date, b, e)

        int_id = tf.cast(id_, tf.int64)
        features = tf.cast(movement_features, tf.float32)
        features, timestamps, time_ranges_tensor, id_ = tf.py_func(
            _xform, 
            [features, int_id],
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


