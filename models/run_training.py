import os
import json
import math
import layers
import utility
import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics
from tensorflow.python.client import timeline

""" TODO(alexwilson):
  9. Build inference application.
  11. Vessel labels as time series also for change of behaviour.
"""

class ModelConfiguration(object):
  """ Configuration for the vessel behaviour model, shared between training and
      inference.
  """

  def __init__(self):
    self.feature_duration_days = 180
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
    self.min_viable_timeslice_length = 500

class ModelTrainer(ModelConfiguration):
  """ Handles the mechanics of training and evaluating a vessel behaviour
      model.
  """ 

  def __init__(self, base_feature_path, train_scratch_path):
    ModelBase.__init__(self)

    self.base_feature_path = base_feature_path
    self.train_scratch_path = train_scratch_path
    self.checkpoint_dir = self.train_scratch_path + '/train'
    self.eval_dir = self.train_scratch_path + '/eval'
    self.num_parallel_readers = 24

  def _training_data_reader(self, input_file_pattern, is_training):
    """ Concurrent training data reader.

    Given a pattern for a set of input files, repeatedly read from these in
    shuffled order, outputing batches of randomly sampled segments of vessel
    tracks for model training or evaluation. Multiple readers are started
    concurrently, and the multiple samples can be output per vessel depending
    upon the weight set for each (used for generating more samples for vessel
    types for which we have fewer examples).

    Args:
      input_file_pattern: glob for input files for training.
      is_training: whether the data is for training (or evaluation).

    Returns:
      A tuple of tensors:
        1. A tensor of features of dimension [batch_size, 1, width, depth]
        2. A tensor of int32 labels of dimension [batch_size]
        3. A tensor of one-hot encodings of the labels of dimension
           [batch_size, num_classes]

    """
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
        self.max_window_duration_seconds, self.window_max_points,
        self.min_viable_timeslice_length, max_replication))

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
    """ The function for running a training replica on a worker. """

    input_file_pattern = self.base_feature_path + '/Training/shard-*-of-*.tfrecord'

    features, labels, one_hot_labels = self._training_data_reader(
        input_file_pattern, True)

    logits = layers.misconception_model(features, self.window_size, self.stride,
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
      self.checkpoint_dir,
      master=master,
      is_chief=is_chief,
      number_of_steps=500000,
      save_summaries_secs=30,
      save_interval_secs=60)

  def run_evaluation(self, master):
    """ The function for running model evaluation on the master. """

    input_file_pattern = self.base_feature_path + '/Test/shard-*-of-*.tfrecord'

    features, labels, one_hot_labels = self._training_data_reader(
        input_file_pattern, False)

    logits = layers.misconception_model(features, self.window_size, self.stride,
            self.feature_depth, self.levels, self.num_classes, False)

    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    names_to_values, names_to_updates = metrics.aggregate_metric_map({
        'Test accuracy': metrics.streaming_accuracy(predictions, labels),
        'Test precision': metrics.streaming_precision(predictions, labels),
    })

    # Create the summary ops such that they also print out to std output.
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
        self.checkpoint_dir,
        self.eval_dir,
        num_evals=num_evals,
        eval_op=names_to_updates.values(),
        summary_op=tf.merge_summary(summary_ops),
        eval_interval_secs=120)


def run_training(config, server, trainer):
  if config.is_ps():
    # This task is a parameter server.
    server.join()
  else:
    with tf.Graph().as_default():
      if config.is_worker():
        # This task is a worker, running training and sharing parameters with
        # other workers via the parameter server.
        with tf.device(tf.train.replica_device_setter(
            worker_device = '/job:%s/task:%d' % (config.task_type, config.task_index),
            cluster = config.cluster_spec)):

          # The chief worker is responsible for writing out checkpoints as the
          # run progresses.
          trainer.run_training(server.target, config.is_chief)
      elif config.is_master():
        # This task is the master, running evaluation.
        trainer.run_evaluation(server.target)
      else:
        raise ValueError('Unexpected task type: %s', config.task_type)

def main(args):
  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)

  logging.info("Running with Tensorflow version: %s", tf.__version__)

  trainer = ModelTrainer(args.root_feature_path, args.training_output_path)

  config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  if (config == {}):
    trainer.run_evaluation('localhost')
  else:
    logging.info("Config dictionary: %s", config)

    node_config = ClusterNodeConfig(config)
    server = node_config.server()
    
    run_training(config, server, trainer)

def parse_args():
  """ Parses command-line arguments for training."""
  argparser = argparse.ArgumentParser('Train fishing classification model.')

  argparser.add_argument('--root_feature_path', default=None,
      help='The root path to the vessel movement feature directories.')

  argparser.add_argument('--training_output_path', default=None,
      help='The working path for model statistics and checkpoints.')

  return argparser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  main(args)
