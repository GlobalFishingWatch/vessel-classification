from __future__ import absolute_import

import argparse
import datetime
from models.alex import layers
import logging
import pytz
import tensorflow.contrib.slim as slim
import tensorflow as tf
import time
import utility


class ModelInference:

    def __init__(self, model, model_checkpoint_path, unclassified_feature_path):
        utility.ModelConfiguration.__init__(self)

        self.model = model
        self.model_checkpoint_path = model_checkpoint_path
        self.unclassified_feature_path = unclassified_feature_path
        self.batch_size = 64
        self.min_points_for_classification = 250

        def _build_starts():
            today = datetime.datetime.now(pytz.utc)
            months = [1, 4, 7, 10]
            year = 2012
            time_starts = []
            while True:
                for month in months:
                    dt = datetime.datetime(year, month, 1, tzinfo=pytz.utc)
                    time_starts.append(int(time.mktime(dt.timetuple())))
                    if dt > today:
                        return time_starts
                year += 1

        time_starts = _build_starts()

        self.time_ranges = [(s, e)
                            for (s, e) in zip(time_starts, time_starts[2:])]

    def run_inference(self, inference_parallelism, inference_results_path):
        matching_files_i = tf.matching_files(self.unclassified_feature_path)
        matching_files = tf.Print(matching_files_i, [matching_files_i],
                                  "Files: ")
        filename_queue = tf.train.input_producer(
            matching_files, shuffle=False, num_epochs=1)

        readers = []
        for _ in range(inference_parallelism):
            reader = utility.cropping_all_slice_feature_file_reader(
                filename_queue, self.num_feature_dimensions + 1,
                self.time_ranges, self.window_max_points,
                self.min_points_for_classification)
            readers.append(reader)

        features, time_ranges, mmsis = tf.train.batch_join(
            readers,
            self.batch_size,
            enqueue_many=True,
            capacity=1000,
            shapes=[[1, self.window_max_points, self.num_feature_dimensions],
                    [2], []])

        logits = self.model.build_inference_net(features)

        softmax = slim.softmax(logits)

        predictions = tf.cast(tf.argmax(softmax, 1), tf.int32)
        max_probabilities = tf.reduce_max(softmax, [1])

        # Open output file, on cloud storage - so what file api?
        config = tf.ConfigProto(
            inter_op_parallelism_threads=inference_parallelism,
            intra_op_parallelism_threads=inference_parallelism)
        with tf.Session(config=config) as sess:
            init_op = tf.group(tf.initialize_local_variables(),
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
            i = 0
            with open(inference_results_path, 'w') as output_file:
                while True:
                    logging.info("Inference step: %d", i)
                    i += 1
                    result = sess.run(
                        [mmsis, time_ranges, predictions, max_probabilities])
                    for mmsi, (
                            start_time_seconds,
                            end_time_seconds), label, max_probability in zip(
                                *result):
                        start_time = datetime.datetime.utcfromtimestamp(
                            start_time_seconds)
                        end_time = datetime.datetime.utcfromtimestamp(
                            end_time_seconds)
                        output_file.write('%d, %s, %s, %s, %.3f\n' % (
                            mmsi, start_time.isoformat(), end_time.isoformat(),
                            utility.VESSEL_CLASS_NAMES[label],
                            max_probability))

            # Write predictions to file: mmsi, max_feature, logits.


def main(args):
    logging.getLogger().setLevel(logging.DEBUG)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model_checkpoint_path = args.model_checkpoint_path
    unclassified_feature_path = args.unclassified_feature_path
    inference_results_path = args.inference_results_path
    inference_parallelism = args.inference_parallelism

    inference = ModelInference(model_checkpoint_path,
                               unclassified_feature_path)
    inference.run_inference(inference_parallelism, inference_results_path)


def parse_args():
    """ Parses command-line arguments for training."""
    argparser = argparse.ArgumentParser(
        'Infer behavioural labels for a set of vessels.')

    argparser.add_argument(
        '--unclassified_feature_path',
        required=True,
        help='The path to the unclassified vessel movement feature directories.')

    argparser.add_argument(
        '--model_checkpoint_path',
        required=True,
        help='Path to the checkpointed model to use for inference.')

    argparser.add_argument(
        '--inference_results_path',
        required=True,
        help='Path to the csv file to dump all inference results.')

    argparser.add_argument(
        '--inference_parallelism',
        type=int,
        default=4,
        help='Path to the csv file to dump all inference results.')

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
