import argparse
import datetime
import layers
import logging
import tensorflow.contrib.slim as slim
import tensorflow as tf
import utility

class ModelInference(utility.ModelConfiguration):
  def __init__(self, model_checkpoint_path, unclassified_feature_path):
    utility.ModelConfiguration.__init__(self)

    self.model_checkpoint_path = model_checkpoint_path
    self.unclassified_feature_path = unclassified_feature_path


  def run_inference(self, inference_results_path):
    matching_files_i = tf.matching_files(self.unclassified_feature_path)
    matching_files = tf.Print(matching_files_i, [matching_files_i], "Files: ")
    filename_queue = tf.train.input_producer(matching_files, shuffle=False,
        num_epochs = 1)

    # To parallelise. How? Multiple local copies of the graph? Or multiple
    # workers, in which case how do we apportion work appropriately?
    reader = utility.cropping_all_slice_feature_file_reader(filename_queue,
        self.num_feature_dimensions+1, self.max_window_duration_seconds,
        self.window_max_points)
    features, time_ranges, mmsis = tf.train.batch(reader, self.batch_size,
      enqueue_many=True, allow_smaller_final_batch=True, capacity=1000,
      shapes=[[1, self.window_max_points, self.num_feature_dimensions], [2], []])

    features = self.zero_pad_features(features)

    logits = layers.misconception_model(features, self.window_size, self.stride,
            self.feature_depth, self.levels, self.num_classes, False)

    softmax = slim.softmax(logits)

    predictions = tf.cast(tf.argmax(softmax, 1), tf.int32)

    # Open output file, on cloud storage - so what file api?
    parallelism = 16
    config=tf.ConfigProto(
                    inter_op_parallelism_threads=parallelism,
                    intra_op_parallelism_threads=parallelism)
    with tf.Session(config=config) as sess:
      init_op = tf.group(
        tf.initialize_local_variables(),
        tf.initialize_all_variables())

      sess.run(init_op)

      logging.info("Restoring model: %s", self.model_checkpoint_path)
      saver = tf.train.Saver()
      saver.restore(sess, self.model_checkpoint_path)

      logging.info("Starting queue runners.")
      tf.train.start_queue_runners()

      # In a loop, calculate logits and predictions and write out. Will
      # be terminated when an EOF exception is thrown.
      logging.info("Running predictions.")
      while True:
        result = sess.run([mmsis, time_ranges, predictions])
        for mmsi, (start_time_seconds, end_time_seconds), label in zip(*result):
          start_time = datetime.datetime.utcfromtimestamp(start_time_seconds)
          end_time = datetime.datetime.utcfromtimestamp(end_time_seconds)
          logging.info("%d, %s, %s, %s", mmsi, start_time.isoformat(), end_time.isoformat(), label)

      # Write predictions to file: mmsi, max_feature, logits.

def main(args):
  logging.getLogger().setLevel(logging.DEBUG)
  tf.logging.set_verbosity(tf.logging.DEBUG)

  model_checkpoint_path = args.model_checkpoint_path
  unclassified_feature_path = args.unclassified_feature_path
  inference_results_path = args.inference_results_path

  inference = ModelInference(model_checkpoint_path, unclassified_feature_path)
  inference.run_inference(inference_results_path)

def parse_args():
  """ Parses command-line arguments for training."""
  argparser = argparse.ArgumentParser('Infer behavioural labels for a set of vessels.')

  argparser.add_argument('--unclassified_feature_path', required=True,
      help='The path to the unclassified vessel movement feature directories.')

  argparser.add_argument('--model_checkpoint_path', required=True,
      help='Path to the checkpointed model to use for inference.')

  argparser.add_argument('--inference_results_path', required=True,
      help='Path to the csv file to dump all inference results.')

  return argparser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  main(args)
