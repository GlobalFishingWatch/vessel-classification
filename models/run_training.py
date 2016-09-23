import os
import json
import math
import utility
import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics
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
    # We allocate a much smaller buffer than would fit the specified time
    # sampled at 5 mins intervals, on the basis that the sample is almost
    # always much more sparse.
    self.window_max_points = (self.max_window_duration_seconds /
        self.max_sample_frequency_seconds) / 4
    self.window_size = 3
    self.stride = 2
    self.feature_depth = 20
    self.levels = 10
    self.batch_size = 32
    self.num_parallel_readers = 24

  def data_reader(self, input_file_pattern, is_training):
    matching_files_i = tf.matching_files(input_file_pattern)
    matching_files = tf.Print(matching_files_i, [matching_files_i], "Files: ")
    filename_queue = tf.train.input_producer(matching_files, shuffle=True)
    capacity = 1000
    min_size_after_deque = capacity - self.batch_size * 4

    max_replication = 8 if is_training else 1

    readers = []
    for _ in range(self.num_parallel_readers):
      readers.append(utility.cropping_weight_replicating_feature_file_reader(
        filename_queue, self.num_feature_dimensions + 1,
        self.max_window_duration_seconds, self.window_max_points, max_replication))

    raw_features, time_bounds, labels = tf.train.shuffle_batch_join(readers,
        self.batch_size, capacity,
        min_size_after_deque,
        enqueue_many=True,
        shapes=[[1, self.window_max_points, self.num_feature_dimensions], [2], []])

    feature_pad_size = self.feature_depth - self.num_feature_dimensions
    assert(feature_pad_size >= 0)
    zero_padding = tf.zeros([self.batch_size, 1, self.window_max_points, feature_pad_size])
    features = tf.concat(3, [raw_features, zero_padding])

    one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)

    return features, labels, one_hot_labels

  def run_training(self, master, is_chief):
    input_file_pattern = self.base_feature_path + '/Training/shard-*-of-*.tfrecord'

    features, labels, one_hot_labels = self.data_reader(input_file_pattern, True)

    logits = utility.misconception_model(features, self.window_size, self.stride,
            self.feature_depth, self.levels, self.num_classes, True)

    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    tf.scalar_summary('Training loss', loss)

    accuracy = slim.metrics.accuracy(labels, predictions)
    tf.scalar_summary('Training accuracy', accuracy)

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = slim.learning.create_train_op(loss, optimizer,
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    model_variables = slim.get_model_variables()

    slim.learning.train(
      train_op,
      self.train_scratch_path + '/train',
      master=master,
      is_chief=is_chief,
      number_of_steps=500000,
      save_summaries_secs=30,
      save_interval_secs=60)

  def run_evaluation(self, master):
    input_file_pattern = self.base_feature_path + '/Test/shard-*-of-*.tfrecord'
    checkpoint_dir = self.train_scratch_path + '/train'
    output_dir = self.train_scratch_path + '/eval'

    features, labels, one_hot_labels = self.data_reader(input_file_pattern, False)

    logits = utility.misconception_model(features, self.window_size, self.stride,
            self.feature_depth, self.levels, self.num_classes, False)

    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    names_to_values, names_to_updates = metrics.aggregate_metric_map({
        'Test accuracy': metrics.streaming_accuracy(predictions, labels),
        'Test precision': metrics.streaming_precision(predictions, labels),
    })

    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for metric_name, metric_value in names_to_values.iteritems():
      op = tf.scalar_summary(metric_name, metric_value)
      op = tf.Print(op, [metric_value], metric_name)
      summary_ops.append(op)
    
    num_examples = 256
    num_evals = math.ceil(num_examples / float(self.batch_size))

    # Setup the global step.
    slim.get_or_create_global_step()

    slim.evaluation.evaluation_loop(
        master,
        checkpoint_dir,
        output_dir,
        num_evals=num_evals,
        eval_op=names_to_updates.values(),
        summary_op=tf.merge_summary(summary_ops),
        eval_interval_secs=120)

def run():
  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)

  logging.info("Running with Tensorflow version: %s", tf.__version__)

  base_feature_path = 'gs://alex-dataflow-scratch/features-scratch/20160922T075356Z'
  train_scratch_path = 'gs://alex-dataflow-scratch/cloudml/model-train-scratch-eval-simple-weighted-correct'
  feature_duration_days = 180
  trainer = Trainer(base_feature_path, train_scratch_path, feature_duration_days)

  config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  if (config == {}):
    trainer.run_evaluation('localhost')
  else:
    cluster_spec = config['cluster']
    task_spec = config['task']
    task_type = task_spec['type']
    task_index = task_spec['index']
    logging.info("Config dictionary: %s", config)

    server = tf.train.Server(cluster_spec,
                             job_name=task_type,
                             task_index=task_index)

    logging.info("Server target: %s", server.target)

    if task_type == 'ps':
      server.join()
    else:
      with tf.Graph().as_default():
        if task_type == 'worker':
          with tf.device(tf.train.replica_device_setter(
              worker_device="/job:%s/task:%d" % (task_type, task_index), cluster=cluster_spec)):
            is_chief = task_index == 0
            trainer.run_training(server.target, is_chief)
        elif task_type == 'master':
          trainer.run_evaluation(server.target)
        else:
          raise ValueError('Unexpected task type: %s', task_type)

if __name__ == '__main__':
  run()
