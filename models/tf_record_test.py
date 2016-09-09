import os

import logging
import tensorflow as tf
#import tensorflow.contrib.slim as slim

""" TODO(alexwilson):

  0. Reading data from GCS?
  1. Deserialise sequence data.
  2. Slice data randomly into three-month sections based on the timestamp field,
     then remove the timestamp field as not a valid feature.
  3. Pad three-month data into a fixed length vector (every 5 mins max, so ~26k).
  4. Split code into training and test pipelines.
  5. Ensure appropriate queues everywhere to allow decent batching.
  6. Build an initial Inception-equivalent model.
  7. Get it running.
  8. Get it running on GPU.
  9. Build inference application.
  10. Investigate Cloud ML.
  11. Vessel labels as time series also for change of behaviour.

  Slicing:
    - tf.random_crop to extract a slice that could contain a max of 3 months.
    - First timestamp (in crop) + 90 * 24 * 3600 as upper limit.
    - While loop with counter. Copy the value at that index through. If the 
      timestamp exceeds the upper time limit, reset counter to zero else inc.

"""

def test():
  limit = 33
  with tf.Graph().as_default():
    test_dims = 20

    input_range = tf.cast(tf.range(0, test_dims), tf.float32)
    packed_range = tf.pack([input_range], axis=1)
    random = tf.random_uniform([test_dims, 6])

    mixed = tf.concat(1, [packed_range, random])
    
    start = tf.constant([[]], shape=[0, 7])

    def update(i, index, t):
      element = tf.gather(mixed, index)
      element_timestamp = tf.gather(element, 0)
      index_inc = tf.add(index, 1)
      next_index = tf.cond(tf.less(element_timestamp, 7), lambda: index_inc, lambda: tf.constant(0))
      return [tf.add(i, 1), next_index, tf.concat(0, [t, tf.pack([element])])]

    def condition(i, index, t):
      return tf.less(i, test_dims)

    #res = tf.concat(1, [start, inc])
    init = tf.initialize_all_variables()

    res = tf.while_loop(condition, update, [tf.constant(0), tf.constant(0), start])
    

    sess = tf.Session()
    sess.run(init)
    print(sess.run(mixed))
    
    #print(sess.run(random))
    print(sess.run(res))



def run():
  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)
  with tf.Graph().as_default():
    filename = './foo.tfrecord'
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # The serialized example is converted back to actual values.
    # TODO(alexwilson): This is in fact an ExampleSequence not Example, extract
    # the sequence data also.
    context_features, sequence_features = tf.parse_single_sequence_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        context_features = {
            'mmsi': tf.FixedLenFeature([], tf.int64),
            'vessel_type_index': tf.FixedLenFeature([], tf.int64),
            'vessel_type_name': tf.FixedLenFeature([], tf.string),
        },
        sequence_features = {
            'movement_features': tf.FixedLenSequenceFeature(shape=(7,),
                dtype=tf.float32)
        })

    mmsi = tf.cast(context_features['mmsi'], tf.int32)
    type_index = tf.cast(context_features['vessel_type_index'], tf.int32)
    type_name = tf.cast(context_features['vessel_type_name'], tf.string)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(type_name))


if __name__ == '__main__':
  #run()
  test()