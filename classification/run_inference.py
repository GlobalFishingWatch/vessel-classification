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


class Inferer(object):
    def __init__(self, model, model_checkpoint_path, root_feature_path, parallelism=4):

        self.model = model
        self.estimator = model.make_estimator(model_checkpoint_path)
        self.root_feature_path = root_feature_path
        logging.info('created Inferer with Model, %s, and dims %s', model, 
                    model.num_feature_dimensions)
        self.parallelism = 4

    def close(self):
        self.sess.close()


    def _feature_files(self, ids):
        return [
            '%s/%s.tfrecord' % (self.root_feature_path, x)
            for x in ids
        ]

    def _build_time_ranges(self, interval_months, start_date, end_date):
        # TODO: should use min_window_duration here
        window_dur_seconds = self.model.max_window_duration_seconds
        last_viable_date = datetime.now(
            pytz.utc) - timedelta(seconds=window_dur_seconds)
        time_starts = []
        start_year = start_date.year
        month_count = start_date.month - 1
        if start_date.day != 1:
            raise ValueError('start_date must fall on the 1st of the month')
        dt = start_date
        while True:
            year = start_year + month_count // 12
            month = month_count % 12 + 1
            month_count += interval_months
            dt = datetime(year, month, 1, tzinfo=pytz.utc)
            if dt >= end_date:
                break
            time_starts.append(dt)
        delta = timedelta(seconds=self.model.max_window_duration_seconds)
        time_ranges = [(int(time.mktime(dt.timetuple())),
                             int(time.mktime((dt + delta).timetuple())))
                            for dt in time_starts]
        return time_ranges

    def run_inference(self, ids, interval_months, start_date, end_date):
        paths = self._feature_files(ids)

        if self.model.max_window_duration_seconds != 0:
            time_ranges = self._build_time_ranges(interval_months, start_date, end_date)
            input_fn = self.model.make_prediction_input_fn(paths, time_ranges, self.parallelism)
        else:
            input_fn = self.model.make_prediction_input_fn(paths, (start_date, end_date), self.parallelism)

        for result in self.estimator.predict(input_fn=input_fn):

            start_time, end_time = [datetime.utcfromtimestamp(x) for x in result['time_ranges']]
            output = {
                'id': result['id'],
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            for k, v in result.items():
                if k in self.model.objective_map:
                    o = self.model.objective_map[k]
                    output[o.metadata_label] = o.build_json_results(v, result['timestamps'])

            yield output




