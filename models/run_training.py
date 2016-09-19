import os

import utility
import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline

""" TODO(alexwilson):
  5. Ensure appropriate queues everywhere to allow decent batching.
  6. Build an initial Inception-equivalent model.
  7. Get it running.
  8. Get it running on GPU.
  9. Build inference application.
  10. Investigate Cloud ML.
  11. Vessel labels as time series also for change of behaviour.
"""

MAX_SAMPLE_FREQUENCY_SECONDS = 5 * 60
NUM_FEATURE_DIMENSIONS = 9
NUM_CLASSES = 9

def run_training(base_feature_path, logdir, feature_duration_days):
    max_window_duration_seconds = feature_duration_days * 24 * 3600
    window_max_points = max_window_duration_seconds / MAX_SAMPLE_FREQUENCY_SECONDS
    window_size = 3
    stride = 2
    feature_depth = 20
    levels = 12

    input_file_pattern = base_feature_path + '/Training/shard-*-of-*.tfrecord'
    num_parallel_readers = 8
    batch_size = 32

    matching_files_i = tf.matching_files(input_file_pattern)
    matching_files = tf.Print(matching_files_i, [matching_files_i], "Files: ")
    filename_queue = tf.train.input_producer(matching_files, shuffle=True)
    capacity = batch_size * 16
    min_size_after_deque = batch_size * 12

    readers = []
    for _ in range(num_parallel_readers):
      readers.append(utility.cropping_feature_file_reader(filename_queue,
        NUM_FEATURE_DIMENSIONS + 1, max_window_duration_seconds, window_max_points))

    features, labels = tf.train.shuffle_batch_join(readers, batch_size, capacity,
        min_size_after_deque,
        shapes=[[1, window_max_points, NUM_FEATURE_DIMENSIONS], []])


    one_hot_labels = slim.one_hot_encoding(labels, NUM_CLASSES)

    feature_pad_size = feature_depth - NUM_FEATURE_DIMENSIONS
    assert(feature_pad_size >= 0)
    zero_padding = tf.zeros([batch_size, 1, window_size, feature_pad_size])
    padded_features = tf.concat(3, [features, zero_padding])

    logits = utility.inception_model(features, window_size, stride,
            feature_depth, levels, NUM_CLASSES)

    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)

    tf.scalar_summary('Total loss', loss)

    optimizer = tf.train.AdamOptimizer(1e-5)
    train_op = slim.learning.create_train_op(loss, optimizer,
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    model_variables = slim.get_model_variables()
    if 0:
      with tf.name_scope('summaries'):
        for v in model_variables:
          tf.histogram_summary(v.name, v)

    slim.learning.train(
      train_op,
      logdir,
      number_of_steps=50000,
      save_summaries_secs=60,
      save_interval_secs=300)


# Parse a cluster spec json with tf.train.ClusterSpec from
#   config = json.loads(os.environ.get('TF_CONFIG', '{}')), see
#   https://cloud.google.com/ml/docs/distributed-training-environ-var
#
# Ps, Workers, master. Latter to run eval.
# How do we get the server into slim for train/eval?

def run():
  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)

  feature_duration_days = 60
  with tf.Graph().as_default():
    run_training('gs://alex-dataflow-scratch/features-scratch/20160917T220846Z',
        'gs://alex-dataflow-scratch/model-train-scratch', feature_duration_days)

if __name__ == '__main__':
  run()
