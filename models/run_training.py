import os
import json

import utility
import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline

""" TODO(alexwilson):
  6. Build an initial Inception-equivalent model.
  9. Build inference application.
  11. Vessel labels as time series also for change of behaviour.
"""

class Trainer(object):
  def __init__(self, base_feature_path, train_scratch_path, feature_duration_days):
    self.base_feature_path = base_feature_path
    self.train_scratch_path = train_scratch_path
    self.num_classes = 9
    self.num_feature_dimensions = 9
    self.max_sample_frequency_seconds = 5 * 60
    self.max_window_duration_seconds = feature_duration_days * 24 * 3600
    self.window_max_points = self.max_window_duration_seconds / self.max_sample_frequency_seconds
    self.window_size = 3
    self.stride = 2
    self.feature_depth = 20
    self.levels = 12
    self.batch_size = 32
    self.num_parallel_readers = 8

  def data_reader(self, input_file_pattern):
    matching_files_i = tf.matching_files(input_file_pattern)
    matching_files = tf.Print(matching_files_i, [matching_files_i], "Files: ")
    filename_queue = tf.train.input_producer(matching_files, shuffle=True)
    capacity = 600
    min_size_after_deque = capacity - self.batch_size * 4

    readers = []
    for _ in range(self.num_parallel_readers):
      readers.append(utility.cropping_feature_file_reader(filename_queue,
        self.num_feature_dimensions + 1, self.max_window_duration_seconds,
        self.window_max_points))

    raw_features, labels = tf.train.shuffle_batch_join(readers, self.batch_size, capacity,
        min_size_after_deque,
        shapes=[[1, self.window_max_points, self.num_feature_dimensions], []])

    feature_pad_size = feature_depth - self.num_feature_dimensions
    assert(feature_pad_size >= 0)
    zero_padding = tf.zeros([self.batch_size, 1, self.window_max_points, feature_pad_size])
    features = tf.concat(3, [raw_features, zero_padding])

    one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)

    return features, labels, one_hot_labels

  def run_training(self, master, is_chief):
    input_file_pattern = self.base_feature_path + '/Training/shard-*-of-*.tfrecord'

    features, labels, one_hot_labels = self.data_reader(input_file_pattern)

    logits = utility.inception_model(features, self.window_size, self.stride,
            self.feature_depth, self.levels, self.num_classes, False)

    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    tf.scalar_summary('Total loss', loss)

    accuracy = slim.metrics.accuracy(labels, predictions)
    tf.scalar_summary('Accuracy', accuracy)

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = slim.learning.create_train_op(loss, optimizer,
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    model_variables = slim.get_model_variables()

    slim.learning.train(
      train_op,
      self.train_scratch_path,
      master=master,
      is_chief=is_chief,
      number_of_steps=50000,
      save_summaries_secs=30,
      save_interval_secs=60)

  def run_evaluation(self):
    input_file_pattern = self.base_feature_path + '/Test/shard-*-of-*.tfrecord'

    features, labels, one_hot_labels = self.data_reader(input_file_pattern)

    logits = utility.inception_model(features, self.window_size, self.stride,
            self.feature_depth, self.levels, self.num_classes, True)

    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'accuracy': slim.metrics.accuracy(predictions, labels),
    })

    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for metric_name, metric_value in metrics_to_values.iteritems():
      op = tf.scalar_summary(metric_name, metric_value)
      op = tf.Print(op, [metric_value], metric_name)
      summary_ops.append(op)

    num_examples = 500
    num_evals = math.ceil(num_examples / float(self.batch_size))

    # Setup the global step.
    slim.get_or_create_global_step()

    slim.evaluation.evaluation_loop(
        'local',
        self.train_scratch_path,
        self.train_scratch_path,
        num_evals=num_evals,
        eval_op=names_to_updates.values(),
        summary_op=tf.merge_summary(summary_ops),
        eval_interval_secs=eval_interval_secs)

def run():
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

  base_feature_path = 'gs://alex-dataflow-scratch/features-scratch/20160917T220846Z'
  train_scratch_path = 'gs://alex-dataflow-scratch/model-train-scratch-eval'
  feature_duration_days = 60
  trainer = Trainer(base_feature_path, train_scratch_path, feature_duration_days)

  
  # We run a separate training coordinator on each worker.
  # TODO(alexwilson): This can't be the best way to pass the local master
  #   address in? Surely we must be able to pull it out of 'server'?
  master = 'grpc://' + cluster_spec['worker'][task_index]

  with tf.Graph().as_default():
    if task_type == 'ps':
      server.join()
    else:
      with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % task_index, cluster=cluster_spec)):
        if task_type == 'worker':
          is_chief = task_index == 0
          trainer.run_training(master, is_chief)
        elif task_type == 'master':
          server.join()
          trainer.run_evaluation()
        else:
          logging.error('Unexpected task type: %s', task_type)
          sys.exit(-1)

if __name__ == '__main__':
  run()
