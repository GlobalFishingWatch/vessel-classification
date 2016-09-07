import os

import logging
import tensorflow as tf

def run():
  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)
  with tf.Graph().as_default():
    filename = './foo.tfrecord'
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features = {
            'mmsi': tf.FixedLenFeature([], tf.int64),
            'vessel_type_index': tf.FixedLenFeature([], tf.int64),
            'vessel_type_name': tf.FixedLenFeature([], tf.string),
        })

    mmsi = tf.cast(features['mmsi'], tf.int32)
    type_index = tf.cast(features['vessel_type_index'], tf.int32)
    type_name = tf.cast(features['vessel_type_name'], tf.string)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(type_name))


if __name__ == '__main__':
  run()