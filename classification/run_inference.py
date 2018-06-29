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

import logging
import numpy as np
import os
import pytz
import subprocess
import tempfile
import tensorflow as tf
import time
import uuid
from datetime import datetime
from datetime import timedelta

from . import file_iterator


class Inferer(object):
    def __init__(self, model, model_checkpoint_path, root_feature_path):

        self.model = model
        self.model_checkpoint_path = model_checkpoint_path
        self.root_feature_path = root_feature_path
        self.batch_size = self.model.batch_size
        self.min_points_for_classification = model.min_viable_timeslice_length
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
        # with self.sess.as_default():
            self.features_ph = tf.placeholder(tf.float32, 
                shape=[None, 1, self.model.window_max_points, self.model.num_feature_dimensions])
            self.timestamps_ph = tf.placeholder(tf.int32, shape=[None, self.model.window_max_points])
            self.time_ranges_ph = tf.placeholder(tf.int32, shape=[None, 2])
            self.mmsis_ph = tf.placeholder(tf.int64, shape=[None])  # TODO: MMSI_CLEANUP -> tf.string
            objectives = self.model.build_inference_net(self.features_ph, self.timestamps_ph,
                                                        self.time_ranges_ph)
            return objectives

    def _restore_graph(self):
        init_op = tf.group(tf.local_variables_initializer(),
                           tf.global_variables_initializer())

        self.sess.run(init_op)
        logging.info("Restoring model: %s", self.model_checkpoint_path)
        saver = tf.train.Saver()
        gspath = self.model_checkpoint_path.startswith('gs:')
        # Models over a certain size don't seem to load properly from gcs(?!),
        # so copy locally and then load
        if gspath:
            tempdir = tempfile.gettempdir()
            temppath = os.path.join(tempdir, uuid.uuid4().hex)
            subprocess.check_call(['gsutil', 'cp', self.model_checkpoint_path, temppath])
            model_checkpoint_path = temppath
        else:
            model_checkpoint_path = self.model_checkpoint_path
        try:
            saver.restore(self.sess, model_checkpoint_path)
        finally:
            os.unlink(temppath)

    def _feature_files(self, mmsis):
        return [
            '%s/%s.tfrecord' % (self.root_feature_path, mmsi)
            for mmsi in mmsis
        ]

    def _build_starts(self, interval_months):
        # TODO: should use min_window_duration here
        window_dur_seconds = self.model.max_window_duration_seconds
        last_viable_date = datetime.now(
            pytz.utc) - timedelta(seconds=window_dur_seconds)
        time_starts = []
        start_year = 2012
        month_count = 0
        while True:
            year = start_year + month_count // 12
            month = month_count % 12 + 1
            month_count += interval_months
            dt = datetime(year, month, 1, tzinfo=pytz.utc)
            if dt > last_viable_date:
                break
            else:
                time_starts.append(dt)
        return time_starts



    def run_inference(self, mmsis, interval_months, start_date, end_date):
        matching_files = self._feature_files(mmsis)
        logging.info("MATCHING:")
        for path in matching_files:
            logging.info("matching_files: %s", path)
        # filename_queue = tf.train.input_producer(
        #     matching_files, shuffle=False, num_epochs=1)



        if self.model.max_window_duration_seconds != 0:

            time_starts = self._build_starts(interval_months)

            delta = timedelta(
                seconds=self.model.max_window_duration_seconds)
            self.time_ranges = [(int(time.mktime(dt.timetuple())),
                                 int(time.mktime((dt + delta).timetuple())))
                                for dt in time_starts]
            feature_iter = file_iterator.cropping_all_slice_feature_file_iterator(
                matching_files, self.deserializer,
                self.time_ranges, self.model.window_max_points,
                self.min_points_for_classification)  # TODO: add year
        else:
            if self.model.window is None:
                shift = self.model.window_max_points
            else:
                b, e = self.model.window
                shift = e - b
            logging.info("Shift %s %s %s", start_date, end_date, shift)
            feature_iter = file_iterator.all_fixed_window_feature_file_iterator(
                matching_files, self.deserializer,
                self.model.window_max_points, shift, start_date, end_date, b, e)


        objectives = self.objectives

        all_predictions = [o.prediction for o in objectives]

        # In a loop, calculate logits and predictions and write out. Will
        # be terminated when an EOF exception is thrown.
        for i, queue_vals in enumerate(feature_iter):
            logging.info("Inference step: %d", i)
            t0 = time.clock()
            logging.info("Type of queue_vals: %s", type(queue_vals))
            for i, qv in enumerate(queue_vals):
                logging.info("type(QV[%s]) = %s", i, type(qv))
                logging.info("tf.shape(QV[%s]) = %s", i, np.shape(qv))
            logging.info("Type of queue_vals: %s", type(queue_vals))

            feed_dict = {
                self.features_ph : queue_vals[0][np.newaxis],
                self.timestamps_ph : queue_vals[1][np.newaxis],
                self.time_ranges_ph : queue_vals[2][np.newaxis],
                self.mmsis_ph : queue_vals[3][np.newaxis]
            }

            batch_result = self.sess.run([self.mmsis_ph, self.time_ranges_ph, self.timestamps_ph] 
                                         + all_predictions, 
                                         feed_dict=feed_dict)

            # Tensorflow returns some items one would expect to be shape (1,)
            # as shape (). Compensate for that here by checking for is_scalar
            result = [(x if np.isscalar(x) else x[0]) for x in batch_result]

            mmsi = result[0]
            start_time, end_time = [datetime.utcfromtimestamp(x) for x in result[1]]
            timestamps_array = result[2]
            predictions_array = result[3:]

            output = {
                'mmsi': int(mmsi),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            for (o, p) in zip(objectives, predictions_array):
                output[o.metadata_label] = o.build_json_results(p, timestamps_array)

            yield output




