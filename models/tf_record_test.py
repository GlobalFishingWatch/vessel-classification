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

MAX_SAMPLE_FREQUENCY_SECONDS = 5
NUM_FEATURE_DIMENSIONS = 7
NUM_CLASSES = 9

def run_training(base_feature_path, logdir, feature_duration_days, num_feature_dimensions):
    max_window_duration_seconds = feature_duration_days * 24 * 3600
    window_max_points = max_window_duration_seconds / MAX_SAMPLE_FREQUENCY_SECONDS
    window_size = 3
    stride = 2
    feature_depth = 20
    levels = 11

    input_file_pattern = base_feature_path + '/Training/shard-*-of-*.tfrecord'
    num_parallel_readers = 8
    num_training_epochs = None
    #batch_size = 32
    batch_size = 8

    if 1:

      input_files = tf.matching_files(input_file_pattern)
      filename_queue = tf.train.string_input_producer(input_files, shuffle=True,
          num_epochs=num_training_epochs)

      context, sequence = utility.single_feature_file_reader(filename_queue,
          NUM_FEATURE_DIMENSIONS)

      sess = tf.Session()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      sess.run(tf.initialize_all_variables())

      for i in range(5000):
        logging.info(sess.run(context))

      return 0

    def make_reader(filename_queue):
      return utility.read_and_crop_fixed_window(filename_queue,
          NUM_FEATURE_DIMENSIONS, max_window_duration_seconds,
          window_max_points)

    features, labels = utility.feature_file_reader(input_file_pattern,
            make_reader, num_parallel_readers, NUM_FEATURE_DIMENSIONS, num_training_epochs,
            batch_size)

    one_hot_labels = slim.one_hot_encoding(labels, NUM_CLASSES)

    predictions = utility.inception_model(features, window_size, stride,
            feature_depth, levels, NUM_CLASSES)

    loss = slim.losses.softmax_cross_entropy(predictions, one_hot_labels)

    tf.scalar_summary('Total loss', loss)

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = slim.learning.create_train_op(loss, optimizer,
            update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    if 0:
      sess = tf.Session()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      sess.run(tf.initialize_all_variables())

      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()

      sess.run(loss, options=run_options, run_metadata=run_metadata)

      tl = timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open('timeline.json', 'w') as f:
        f.write(ctf)

      return


    slim.learning.train(
      train_op,
      logdir,
      number_of_steps=5,
      save_summaries_secs=300,
      save_interval_secs=600)



def run():
  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)

  #feature_duration_days = 30
  feature_duration_days = 5
  with tf.Graph().as_default():
    run_training('gs://alex-dataflow-scratch/features-scratch/20160913T200731Z',
        'gs://alex-dataflow-scratch/model-train-scratch', feature_duration_days, 7)

if __name__ == '__main__':
  run()
  #test()
