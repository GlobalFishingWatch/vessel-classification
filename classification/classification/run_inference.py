from __future__ import absolute_import

import argparse
import datetime
import gzip
import importlib
import logging
import newlinejson as nlj
import numpy as np
import os
from pkg_resources import resource_filename
import pytz
import sys
import tensorflow.contrib.slim as slim
import tensorflow as tf
import time
from . import model
from . import params
from . import utility


class Inferer(object):
    def __init__(self, model, model_checkpoint_path, root_feature_path, mmsis):

        self.model = model
        self.model_checkpoint_path = model_checkpoint_path
        self.root_feature_path = root_feature_path
        self.batch_size = self.model.batch_size
        self.min_points_for_classification = model.min_viable_timeslice_length
        self.mmsis = mmsis

    def _feature_files(self, split):
        return [
            '%s/%d.tfrecord' % (self.root_feature_path, mmsi)
            for mmsi in self.mmsis
        ]

    def _build_starts(self, interval_length_seconds):
        today = datetime.datetime.now(pytz.utc)
        time_starts = []
        iter = datetime.datetime(2012, 1, 1, tzinfo=pytz.utc)
        while iter < today:
            time_starts.append(int(time.mktime(iter.timetuple())))
            iter += datetime.timedelta(seconds=interval_length_seconds)
        return time_starts

    def run_inference(self, inference_parallelism, inference_results_path):
        matching_files = self._feature_files(self.mmsis)
        filename_queue = tf.train.input_producer(
            matching_files, shuffle=False, num_epochs=1)

        readers = []
        if self.model.max_window_duration_seconds != 0:
            time_starts = self._build_starts(self.model.max_window_duration_seconds)

            self.time_ranges = [(s, e)
                                for (s, e) in zip(time_starts, time_starts[1:])]
            for _ in range(inference_parallelism * 2):
                reader = utility.cropping_all_slice_feature_file_reader(
                    filename_queue, self.model.num_feature_dimensions + 1,
                    self.time_ranges, self.model.window_max_points,
                    self.min_points_for_classification)
                readers.append(reader)
        else:
            for _ in range(inference_parallelism * 2):
                reader = utility.all_fixed_window_feature_file_reader(
                    filename_queue, self.model.num_feature_dimensions + 1,
                    self.model.window_max_points)
                readers.append(reader)

        features, timestamps, time_ranges, mmsis = tf.train.batch_join(
            readers,
            self.batch_size,
            enqueue_many=True,
            capacity=1000,
            shapes=[[
                1, self.model.window_max_points,
                self.model.num_feature_dimensions
            ], [self.model.window_max_points], [2], []])

        objectives = self.model.build_inference_net(features, timestamps,
                                                    mmsis)

        all_predictions = [o.prediction for o in objectives]

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
            with nlj.open(gzip.GzipFile(inference_results_path, 'w'),
                          'w') as output_nlj:
                while True:
                    logging.info("Inference step: %d", i)
                    i += 1
                    batch_results = sess.run([mmsis, time_ranges, timestamps] +
                                             all_predictions)
                    for result in zip(*batch_results):
                        mmsi = result[0]
                        (start_time_seconds, end_time_seconds) = result[1]
                        timestamps_array = result[2]
                        predictions_array = result[3:]

                        start_time = datetime.datetime.utcfromtimestamp(
                            start_time_seconds)
                        end_time = datetime.datetime.utcfromtimestamp(
                            end_time_seconds)

                        output = dict(
                            [(o.metadata_label,
                              o.build_json_results(p, timestamps_array))
                             for (o, p) in zip(objectives, predictions_array)])

                        output.update({
                            'mmsi': int(mmsi),
                            'start_time': start_time.isoformat(),
                            'end_time': end_time.isoformat()
                        })

                        output_nlj.write(output)


def main(args):
    logging.getLogger().setLevel(logging.DEBUG)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model_checkpoint_path = args.model_checkpoint_path
    root_feature_path = args.root_feature_path
    inference_results_path = args.inference_results_path
    inference_parallelism = args.inference_parallelism

    mmsis = utility.find_available_mmsis(args.root_feature_path)

    if args.dataset_split:
        if args.dataset_split in ['Training', 'Test']:
            metadata_file = os.path.abspath(
                resource_filename('classification.data', params.metadata_file))
            fishing_range_file = os.path.abspath(
                resource_filename('classification.data',
                                  params.fishing_ranges_file))
            if not os.path.exists(metadata_file):
                logging.fatal("Could not find metadata file: %s.",
                              metadata_file)
                sys.exit(-1)

            fishing_ranges = utility.read_fishing_ranges(fishing_range_file)
            vessel_metadata = utility.read_vessel_multiclass_metadata(
                mmsis,
                metadata_file,
                fishing_range_dict=fishing_ranges)

            mmsis.intersection_update(
                vessel_metadata.mmsis_for_split(args.dataset_split))
        else:
            mmsis_file = os.path.abspath(
                resource_filename('classification.data', args.dataset_split))
            if not os.path.exists(mmsis_file):
                logging.fatal("Could not find mmsis file: %s.",
                              args.dataset_split)
                sys.exit(-1)
            with open(mmsis_file, 'r') as f:
                mmsis.intersection_update([int(m) for m in f])

    logging.info("Running inference with %d mmsis", len(mmsis))

    module = "classification.models.{}".format(args.model_name)
    try:
        Model = importlib.import_module(module).Model
    except:
        logging.error("Could not load model: {}".format(module))
        raise

    feature_dimensions = int(args.feature_dimensions)
    chosen_model = Model(feature_dimensions, None)

    infererer = Inferer(chosen_model, model_checkpoint_path, root_feature_path,
                        mmsis)
    infererer.run_inference(inference_parallelism, inference_results_path)


def parse_args():
    """ Parses command-line arguments for training."""
    argparser = argparse.ArgumentParser(
        'Infer behavioural labels for a set of vessels.')

    argparser.add_argument('model_name')

    argparser.add_argument(
        '--root_feature_path',
        required=True,
        help='The path to the vessel movement feature directories.')

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

    argparser.add_argument(
        '--dataset_split',
        type=str,
        default='',
        help='Data split to classify. If unspecified, all vessels. Otherwise '
        'if Training or Test, read from built-in training/test split, '
        'otherwise the name of a single-column csv file of mmsis.')

    argparser.add_argument(
        '--feature_dimensions',
        required=True,
        help='The number of dimensions of a classification feature.')

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
