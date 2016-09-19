import os
import json

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

def run_training(master, is_chief, base_feature_path, logdir, feature_duration_days):
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


  logits = utility.inception_model(features, window_size, stride,
          feature_depth, levels, NUM_CLASSES)
  predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

  loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
  tf.scalar_summary('Total loss', loss)

  accuracy = slim.metrics.accuracy(labels, predictions)
  tf.scalar_summary('Accuracy', accuracy)

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
    master=master,
    is_chief=is_chief,
    number_of_steps=50000,
    save_summaries_secs=30,
    save_interval_secs=60)


def run():
  feature_duration_days = 60

  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)

  config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_spec = config['cluster']
  task_spec = config['task']
  task_type = task_spec['type']
  task_index = task_spec['index']
  logging.info("Config dictionary: %s", config)
  server = tf.train.Server(cluster_spec,
                           job_name=task_type,
                           task_index=task_index)

  # We run a separate training coordinator on each worker.
  # TODO(alexwilson): This can't be the best way to pass the local master
  #   address in? Surely we must be able to pull it out of 'server'?
  master = 'grpc://' + cluster_spec['worker'][task_index]

  with tf.Graph().as_default():
    if task_type == 'ps':
      server.join()
    elif task_type == 'worker':
      is_chief = task_index == 0
      with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % task_index, cluster=cluster_spec)):
        run_training(master, is_chief, 'gs://alex-dataflow-scratch/features-scratch/20160917T220846Z',
            'gs://alex-dataflow-scratch/model-train-scratch', feature_duration_days)
    elif task_type == 'master':
      server.join()
    else:
      logging.error('Unexpected task type: %s', task_type)
      sys.exit(-1)      

if __name__ == '__main__':
  run()
