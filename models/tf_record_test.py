import os

import utility
import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim

""" TODO(alexwilson):
  5. Ensure appropriate queues everywhere to allow decent batching.
  6. Build an initial Inception-equivalent model.
  7. Get it running.
  8. Get it running on GPU.
  9. Build inference application.
  10. Investigate Cloud ML.
  11. Vessel labels as time series also for change of behaviour.
"""

NUM_FEATURE_DIMENSIONS = 7
FEATURE_DURATION_DAYS = 30
WINDOW_DURATION_SECONDS = FEATURE_DURATION_DAYS * 24 * 3600
ONE_MONTH_MAX_POINTS = FEATURE_DURATION_DAYS * 24 * (60/5)

def train(features, labels, logdir):
  predictions = utility.inception_model(features, 3, 2, 20, 12)
  loss = slim.losses.softmax_cross_entropy(predictions, labels)
  optimizer = tf.train.AdamOptimizer()
  train_op = slim.learning.create_train_op(loss, optimizer)

  slim.learning.train(
    train_op,
    logdir,
    number_of_steps=1000,
    save_summaries_secs=300,
    save_interval_secs=600)

def run():
  return
  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)
  with tf.Graph().as_default():
    input_file_pattern = 'gs://alex-dataflow-scratch/features-scratch/20160913T200731Z/Training/shard-*-of-*.tfrecord'
    context_features, sequence_features = utility.feature_file_reader(input_file_pattern, NUM_FEATURE_DIMENSIONS)
    movement_features = sequence_features['movement_features']

    cropped_features = utility.random_fixed_time_extract(movement_features, WINDOW_DURATION_SECONDS, ONE_MONTH_MAX_POINTS)

    mmsi = tf.cast(context_features['mmsi'], tf.int32)
    type_index = tf.cast(context_features['vessel_type_index'], tf.int32)
    type_name = tf.cast(context_features['vessel_type_name'], tf.string)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(mmsi))
    print(sess.run(type_name))
    #tf.random_crop
    print(sess.run(tf.shape(cropped_features)))


if __name__ == '__main__':
  run()
  #test()
