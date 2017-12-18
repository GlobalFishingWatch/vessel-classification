# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import tensorflow as tf
import time
from . import model
from . import utility
from . import file_iterator
from itertools import chain
import gc
import subprocess
import resource

def log_dt(t0, message):
    t1 = time.clock()
    logging.info("%s (dt = %s s", message, t1 - t0)
    return t1

def log_mem(message, mmsis):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.info("%s for %s: %s", message, mmsis, mem)

class Inferer(object):
    def __init__(self, model, model_checkpoint_path, root_feature_path,
                       inference_parallelism=1):

        self.model = model
        self.model_checkpoint_path = model_checkpoint_path
        self.root_feature_path = root_feature_path
        self.batch_size = self.model.batch_size
        self.min_points_for_classification = model.min_viable_timeslice_length
        self.inference_parallelism = inference_parallelism
        config = tf.ConfigProto(
            inter_op_parallelism_threads=inference_parallelism,
            intra_op_parallelism_threads=inference_parallelism)
        self.sess = tf.Session()
        self.objectives = self._build_objectives()
        self._restore_graph()
        self.deserializer = file_iterator.Deserializer(
                num_features=model.num_feature_dimensions + 1, sess=self.sess)
        logging.info('created Inferer with Model, %s, and dims %s', model, 
                    model.num_feature_dimensions)

    def close(self):
        self.sess.close()


    def _build_objectives(self):
        with self.sess.as_default():
            t0 = time.clock()

            self.features_ph = tf.placeholder(tf.float32, 
                shape=[None, 1, self.model.window_max_points, self.model.num_feature_dimensions])
            self.timestamps_ph = tf.placeholder(tf.int32, shape=[None, self.model.window_max_points])
            self.time_ranges_ph = tf.placeholder(tf.int32, shape=[None, 2])
            self.mmsis_ph = tf.placeholder(tf.int32, shape=[None])

            t0 = log_dt(t0, "built placeholders")


            objectives = self.model.build_inference_net(self.features_ph, self.timestamps_ph,
                                                        self.time_ranges_ph)

            t0 = log_dt(t0, "built objectives")

            return objectives


    def _restore_graph(self):
        t0 = time.clock()
        init_op = tf.group(tf.local_variables_initializer(),
                           tf.global_variables_initializer())

        self.sess.run(init_op)
        t0 = log_dt(t0, "Initialized variable")
        logging.info("Restoring model: %s", self.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_checkpoint_path)

        t0 = log_dt(t0, "restored net")



    def _feature_files(self, mmsis):
        return [
            '%s/%s.tfrecord' % (self.root_feature_path, mmsi)
            for mmsi in mmsis
        ]

    def _build_starts(self, interval_months):
        # TODO: should use min_window_duration here
        window_dur_seconds = self.model.max_window_duration_seconds
        last_viable_date = datetime.datetime.now(
            pytz.utc) - datetime.timedelta(seconds=window_dur_seconds)
        time_starts = []
        start_year = 2012
        month_count = 0
        while True:
            year = start_year + month_count // 12
            month = month_count % 12 + 1
            month_count += interval_months
            dt = datetime.datetime(year, month, 1, tzinfo=pytz.utc)
            if dt > last_viable_date:
                break
            else:
                time_starts.append(dt)
        return time_starts



    def run_inference(self, mmsis, interval_months, start_date, end_date):
        t0 = time.clock()
        matching_files = self._feature_files(mmsis)
        logging.info("MATCHING:")
        for path in matching_files:
            logging.info("matching_files: %s", path)
        # filename_queue = tf.train.input_producer(
        #     matching_files, shuffle=False, num_epochs=1)



        readers = []
        assert self.inference_parallelism == 1 # TODO: rework
        if self.model.max_window_duration_seconds != 0:

            time_starts = self._build_starts(interval_months)

            delta = datetime.timedelta(
                seconds=self.model.max_window_duration_seconds)
            self.time_ranges = [(int(time.mktime(dt.timetuple())),
                                 int(time.mktime((dt + delta).timetuple())))
                                for dt in time_starts]
            raise NotImplentedError()
            for _ in range(self.inference_parallelism * 2):
                reader = utility.cropping_all_slice_feature_file_reader(
                    filename_queue, self.model.num_feature_dimensions + 1,
                    self.time_ranges, self.model.window_max_points,
                    self.min_points_for_classification)  # TODO: add year
                readers.append(reader)
        else:
            if self.model.window is None:
                shift = self.model.window_max_points
            else:
                b, e = self.model.window
                shift = e - b

            # for _ in range(self.inference_parallelism * 2):
            logging.info("Shift %s %s %s", start_date, end_date, shift)
            reader = file_iterator.all_fixed_window_feature_file_iterator(
                matching_files, self.deserializer,
                self.model.window_max_points, shift, start_date, end_date)
            readers.append(reader)

        t0 = log_dt(t0, "built readers")

        feature_iter = chain(*readers)

        # features, timestamps, time_ranges, mmsis = tf.train.batch_join(
        #     readers,
        #     self.batch_size,
        #     enqueue_many=True,
        #     capacity=1000,
        #     shapes=[[
        #         1, self.model.window_max_points,
        #         self.model.num_feature_dimensions
        #     ], [self.model.window_max_points], [2], []],
        #     allow_smaller_final_batch=True)

        t0 = log_dt(t0, "built queues")

        objectives = self.objectives

        all_predictions = [o.prediction for o in objectives]





        # log_mem("Starting queue runners.", mmsis)
        # resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # threads = tf.train.start_queue_runners(sess=self.sess)

        # In a loop, calculate logits and predictions and write out. Will
        # be terminated when an EOF exception is thrown.
        log_mem("Running predictions.", mmsis)
        i = 0
        while True:
            logging.info("Inference step: %d", i)
            t0 = time.clock()
            i += 1
            try:
                # logging.info("Evaluating queues")
                # queue_vals = self.sess.run([filename_queue])
                # logging.info("Type of queue_vals: %s", type(queue_vals))


                # return

                log_mem("Evaluating queues", mmsis)
                queue_vals = feature_iter.next() # TODO: switch to for loop
                logging.info("Type of queue_vals: %s", type(queue_vals))
                for i, qv in enumerate(queue_vals):
                    logging.info("type(QV[%s]) = %s", i, type(qv))
                    logging.info("tf.shape(QV[%s]) = %s", i, np.shape(qv))
                logging.info("Type of queue_vals: %s", type(queue_vals))
                feed_dict = {
                    self.features_ph : [queue_vals[0]],
                    self.timestamps_ph : [queue_vals[1]],
                    self.time_ranges_ph : [queue_vals[2]],
                    self.mmsis_ph : [queue_vals[3]]
                }
                t0 = log_dt(t0, "Queues evaluated")
                log_mem("Queues evaluated", mmsis)
                logging.info("GC count: %s", gc.get_count())
                logging.info("DF: %s", subprocess.check_output('df'))
            except StopIteration as err:
                logging.info("Queues exhausted")
                break
            log_mem("Running Session", mmsis)
            batch_results = self.sess.run(all_predictions, feed_dict=feed_dict)
            log_mem("Ran Session", mmsis)
            t0 = log_dt(t0, "executed step")
            logging.info("queue_vals type %s, len %s", type(queue_vals), len(queue_vals))
            for qv, predictions_array in zip([queue_vals], batch_results):
                mmsi = qv[3]
                (start_time_seconds, end_time_seconds) = qv[2]
                timestamps_array = qv[1]

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
                t0 = log_dt(t0, "created output")

                yield output


def main(args):
    logging.getLogger().setLevel(logging.DEBUG)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model_checkpoint_path = args.model_checkpoint_path
    root_feature_path = args.root_feature_path
    inference_results_path = args.inference_results_path
    inference_parallelism = args.inference_parallelism

    mmsis = utility.find_available_mmsis(args.root_feature_path)

    module = "classification.models.{}".format(args.model_name)
    try:
        Model = importlib.import_module(module).Model
    except:
        logging.error("Could not load model: {}".format(module))
        raise

    if args.dataset_split:
        if args.dataset_split in ['Training', 'Test']:
            metadata_file = os.path.abspath(
                resource_filename('classification.data', args.metadata_file))
            fishing_range_file = os.path.abspath(
                resource_filename('classification.data',
                                  args.fishing_ranges_file))
            if not os.path.exists(metadata_file):
                logging.fatal("Could not find metadata file: %s.",
                              metadata_file)
                sys.exit(-1)

            fishing_ranges = utility.read_fishing_ranges(fishing_range_file)
            vessel_metadata = Model.read_metadata(
                mmsis, metadata_file, fishing_ranges=fishing_ranges)

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
    if args.mmsi:
        if args.dataset_split:
            logging.fatal("Only one of `mmsi` or `dataset_split` can be specified")
            sys.exit(-1)
        mmsis = [int(args.mmsi)]

    logging.info("Running inference with %d mmsis", len(mmsis))

    feature_dimensions = int(args.feature_dimensions)
    chosen_model = Model(feature_dimensions, None, None)

    infererer = Inferer(chosen_model, model_checkpoint_path, root_feature_path,
                        inference_parallelism)

    if args.interval_months is None:
        # This is ignored for point inference, but we can't care.
        interval_months = 6
    else:
        # Break if the user sets a time interval when we can't honor it.
        assert chosen_model.max_window_duration_seconds != 0, "can't set interval for point inferring model"
        interval_months = args.interval_months


    with nlj.open(gzip.GzipFile(inference_results_path, 'w'),
                          'w') as output_nlj:
        for x in infererer.run_inference(mmsis,
                                interval_months, args.start_date, args.end_date):
            output_nlj.write(x)


def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=pytz.utc)
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


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
        '--mmsi',
        type=str,
        default='',
        help='Run inference only for the given MMSI. Not compatible with'
             'dataset_split.')

    argparser.add_argument(
        '--feature_dimensions',
        required=True,
        help='The number of dimensions of a classification feature.')

    argparser.add_argument(
        '--metadata_file',
        required=True,
        help='Name of file containing metadata.')

    argparser.add_argument(
        '--fishing_ranges_file',
        required=True,
        help='Name of the file containing fishing ranges.')

    argparser.add_argument(
        '--interval_months',
        default=None,
        type=int,
        help="Interval between successive classifications")

    argparser.add_argument(
        '--start_date',
        default=None,
        type=valid_date,
        help='start of period to run inference on (defaults to earliest date with data)')

    argparser.add_argument(
        '--end_date',
        default=None,
        type=valid_date,
        help='stop of period to run inference on (defaults to latest date with data)')

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
