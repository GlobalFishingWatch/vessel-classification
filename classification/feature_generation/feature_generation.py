import tensorflow as tf
import numpy as np


def filename_generator(filenames, random_state):
    while True:
        yield random_state.choice(filenames)


def flatten_features(features, timestamps, time_ranges, mmsi):
    return tf.data.Dataset.from_tensor_slices((features, timestamps, time_ranges, mmsi))


def set_feature_shapes(all_features, num_features, window_size):
    features, timestamps, time_ranges, mmsi = all_features
    features.set_shape([window_size, num_features - 1]) 
    timestamps.set_shape([window_size])
    time_ranges.set_shape([2])
    mmsi.set_shape([])


def parse_function_core(example_proto, num_features):
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


def read_input_fn_infinite(paths, num_features, num_parallel_reads=4, random_state=None):
    
    def parse_function(example_proto):
        return parse_function_core(example_proto, num_features)
    
    if random_state is None:
        random_state = np.random.RandomState()

    path_ds = tf.data.Dataset.from_generator(lambda:filename_generator(paths, random_state), tf.string)

    return (tf.data.TFRecordDataset(path_ds, num_parallel_reads=num_parallel_reads)
                .prefetch(num_parallel_reads)     
                .map(parse_function, num_parallel_calls=num_parallel_reads)
           )


def read_input_fn_one_shot(paths, num_features, num_parallel_reads=4):
    
    def parse_function(example_proto):
        return parse_function_core(example_proto, num_features)

    return (tf.data.TFRecordDataset(paths, num_parallel_reads=num_parallel_reads)
                .prefetch(num_parallel_reads)     
                .map(parse_function, num_parallel_calls=num_parallel_reads)
           )


