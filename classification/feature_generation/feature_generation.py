import tensorflow as tf
import numpy as np
import posixpath as pp
import os

def filename_generator(filenames, random_state, weights):
    if weights is not None:
        weights = np.array(weights)
        weights /= weights.sum()
    while True:
        yield random_state.choice(filenames, p=weights)


def flatten_features(features, timestamps, time_ranges, id_):
    return tf.data.Dataset.from_tensor_slices((features, timestamps, time_ranges, id_))


def set_feature_shapes(all_features, num_features, window_size):
    features, timestamps, time_ranges, id_ = all_features
    features.set_shape([window_size, num_features - 1]) 
    timestamps.set_shape([window_size])
    time_ranges.set_shape([2])
    id_.set_shape([])


def parse_function_core(example_proto, num_features):
    context_features, sequence_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features={
            'id': tf.io.FixedLenFeature([], tf.int64)
            },
        sequence_features={
            'movement_features': tf.io.FixedLenSequenceFeature(shape=(num_features, ), 
                                                            dtype=tf.float32)
        }
    )
    return context_features['id'], sequence_features['movement_features']

def path2id(path):
    return tf.compat.v1.py_func(
            lambda p: pp.splitext(pp.basename(p))[0], [path], tf.string)

def read_input_fn_infinite(paths, num_features, num_parallel_reads=4, 
    random_state=None, weights=None):
    """Read data for training.

    Because we are IO bound during training, we return the raw IDs. These
    are mapped real IDs using the vessel metadata.
    """
    
    def parse_function(example_proto):
        return parse_function_core(example_proto, num_features)
    
    if random_state is None:
        random_state = np.random.RandomState()

    path_ds = tf.data.Dataset.from_generator(lambda:filename_generator(paths, random_state, weights), 
                    tf.string)

    return (tf.data.TFRecordDataset(path_ds, num_parallel_reads=num_parallel_reads)
                .map(parse_function, num_parallel_calls=num_parallel_reads))


def read_input_fn_one_shot(paths, num_features, num_parallel_reads=4):
    """Read data for training.

    Because we are less likely to be IO bound during inference
    we return the real IDs as derived from the filenames.
    """
    
    def parse_function(example_proto):
        return parse_function_core(example_proto, num_features)

    path_ds_1 = tf.data.Dataset.from_tensor_slices(paths)
    path_ds_2 = tf.data.Dataset.from_tensor_slices(paths)

    return tf.data.Dataset.zip((
        path_ds_1
            .map(path2id),
        tf.data.TFRecordDataset(path_ds_2)
            .map(parse_function)
            .map(lambda id_, features: features)
        ))


