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
    return tf.py_function(
            lambda p: pp.splitext(pp.basename(p.numpy()))[0], [path], tf.string)

# dataset = filenames.flat_map(
#     lambda fn: tf.data.Dataset.zip((tf.data.Dataset.from_tensors(fn).repeat(None),
#                                     tf.data.TFRecordDataset(fn))))

# def read_input_fn_infinite(paths, num_features, num_parallel_reads=4, 
#     random_state=None, weights=None):
    
#     def parse_function(example_proto):
#         return parse_function_core(example_proto, num_features)
    
#     if random_state is None:
#         random_state = np.random.RandomState()

#     path_ds = tf.data.Dataset.from_generator(lambda:filename_generator(paths, random_state, weights), 
#                     tf.string)

#     return tf.data.Dataset.zip((
#         path_ds
#             .prefetch(num_parallel_reads)
#             .map(path2id, num_parallel_calls=num_parallel_reads),
#         tf.data.TFRecordDataset(path_ds, num_parallel_reads=num_parallel_reads)
#             .prefetch(num_parallel_reads)
#             .map(parse_function, num_parallel_calls=num_parallel_reads)
#         ))



# def read_input_fn_infinite(paths, num_features, num_parallel_reads=4, 
#     random_state=None, weights=None):
    
#     def parse_function(fn, example_proto):
#         return path2id(fn), parse_function_core(example_proto, num_features)
    
#     if random_state is None:
#         random_state = np.random.RandomState()

#     path_ds = tf.data.Dataset.from_generator(lambda:filename_generator(paths, random_state, weights), 
#                     tf.string)

#     dataset = path_ds.flat_map(
#         lambda fn: tf.data.Dataset.zip((tf.data.Dataset.from_tensors(fn).repeat(None),
#                                         tf.data.TFRecordDataset(fn, 
#                                                 num_parallel_reads=num_parallel_reads))))

#     return dataset.map(parse_function, num_parallel_calls=num_parallel_reads)


#     return tf.data.Dataset.zip((
#         path_ds
#             .prefetch(num_parallel_reads)
#             .map(path2id, num_parallel_calls=num_parallel_reads),
#             # Can't use multiple readers here since the results are returned out of order
#         tf.data.TFRecordDataset(path_ds, num_parallel_reads=1)
#             .prefetch(num_parallel_reads)
#             .map(parse_function, num_parallel_calls=num_parallel_reads)
#         ))

# def read_input_fn_infinite(paths, num_features, num_parallel_reads=4, 
#     random_state=None, weights=None):
    
#     def parse_function(fn, example_proto):
#         return path2id(fn), parse_function_core(example_proto, num_features)
    
#     if random_state is None:
#         random_state = np.random.RandomState()

#     path_ds = tf.data.Dataset.from_generator(lambda:filename_generator(paths, random_state, weights), 
#                     tf.string)

#     dataset = path_ds.interleave(
#         lambda fn: tf.data.Dataset.zip((tf.data.Dataset.from_tensors(fn).repeat(None),
#                                         tf.data.TFRecordDataset(fn, 
#                                                 num_parallel_reads=num_parallel_reads))),
#                 cycle_length=8, block_length=1)

#     return dataset.map(parse_function, num_parallel_calls=num_parallel_reads)
    # return tf.data.Dataset.zip((
    #     path_ds
    #         .prefetch(num_parallel_reads)
    #         .map(path2id, num_parallel_calls=num_parallel_reads),
    #         # Can't use multiple readers here since the results are returned out of order
    #     tf.data.TFRecordDataset(path_ds, num_parallel_reads=1)
    #         .prefetch(num_parallel_reads)
    #         .map(parse_function, num_parallel_calls=num_parallel_reads)
    #     ))


# TODO: include real ids in the training data.
# 1. Add `id_bytes` field to training data that has the real ids (in sharding)
# 2. Once all features are regenerated, switch these routines to just pull ID bytes
#    WIll require adaption of X_feature_generation.

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

# def read_input_fn_infinite(paths, num_features, num_parallel_reads=4, 
#     random_state=None, weights=None):
    
#     def parse_function(example_proto):
#         return parse_function_core(example_proto, num_features)
    
#     if random_state is None:
#         random_state = np.random.RandomState()
#     random_state_2 = np.random.RandomState()
#     random_state_2.set_state(random_state.get_state())

#     path_ds_1 = tf.data.Dataset.from_generator(
#                         lambda:filename_generator(paths, random_state, weights), tf.string)
#     path_ds_2 = tf.data.Dataset.from_generator(
#                         lambda:filename_generator(paths, random_state_2, weights), tf.string)

#     return tf.data.Dataset.zip((
#         path_ds_1.map(path2id),
#         # We can't use this approach with num_parallel reads > 1 since files are read
#         # in interleaved order.
#         tf.data.TFRecordDataset(path_ds_2, num_parallel_reads=1)
#             .map(parse_function, num_parallel_calls=num_parallel_reads)
#         ))


# TODO: fix tests so they catch the problem with multiple reads with old code

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
            .prefetch(num_parallel_reads)
            .map(path2id, num_parallel_calls=num_parallel_reads),
        tf.data.TFRecordDataset(path_ds_2, num_parallel_reads=1)
            .prefetch(num_parallel_reads)
            .map(parse_function, num_parallel_calls=num_parallel_reads)
            .map(lambda id_, features: features)
        ))


