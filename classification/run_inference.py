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

from feature_generation import file_iterator


class Inferer(object):
    def __init__(self, model, model_checkpoint_path, root_feature_path):

        self.model = model
        self.estimator = model.make_estimator(model_checkpoint_path)

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


    # TODO factor into fishing and vessel inference
    def run_inference(self, mmsis, interval_months, start_date, end_date):
        matching_files = self._feature_files(mmsis)
     
        if self.model.max_window_duration_seconds != 0:
            def input_fn():
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
                return tf.dataset.from_generator(feature_iter).batch(1)
        else:

            def input_fn():
                if self.model.window is None:
                    b, e = 0, self.window_max_points
                else:
                    b, e = self.model.window
                shift = e - b
                feature_iter = file_iterator.all_fixed_window_feature_file_iterator(
                                        matching_files, self.deserializer,
                                        self.model.window_max_points, shift, start_date, end_date, b, e)
                return tf.dataset.Dataset.from_generator(feature_iter).batch(1)

        for batch_result in self.estimator.predict(input_fn=input_fn):


            # Tensorflow returns some items one would expect to be shape (1,)
            # as shape (). Compensate for that here by checking for is_scalar
            # TODO: check if this is still needed with new interface (or maybe can be moved into model_fn when it matters)
            result = {k: (v if np.isscalar(v) else v[0]) for x in batch_result.items()}

            start_time, end_time = [datetime.utcfromtimestamp(x) for x in result[1]]
            output = {
                'mmsi': int(result['mmsi']),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            for k, v in result.items():
                if k in output:
                    continue
                o = self.model.objectives[k]
                output[o.metadata_label] = o.build_json_results(k, result['timestamps'])

            yield output




