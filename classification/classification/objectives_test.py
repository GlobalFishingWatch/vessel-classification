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

import calendar
import dateutil.parser
import numpy as np
import objectives
import utility
import tensorflow as tf


def _dt(s):
    return dateutil.parser.parse(s)


class RegressionLossTest(tf.test.TestCase):
    def test_simple_loss(self):
        with self.test_session():
            prediction = np.array([1.0, 4.0, 5.0, 6.0, 3.0])
            mmsis = np.array([1, 2, 3, 6, 9])

            real_values = {1: 1.0, 2: 4.0, 3: 5.0, 6: 6.0, 9: 3.0}

            objective = objectives.RegressionObjective(
                'a label', 'A name', lambda mmsi: real_values.get(mmsi))

            objective.prediction = prediction
            loss, _ = objective.build_trainer(None, mmsis)

            self.assertAlmostEqual(0.0, loss.eval())

    def test_loss_missing_values(self):
        with self.test_session():
            prediction = np.array([1.0, 4.0, 5.0, 6.0, 3.0])
            mmsis = np.array([1, 2, 3, 4, 5])

            real_values = {1: 2.0, 2: 2.5, 3: 4.5}

            objective = objectives.RegressionObjective(
                'a label', 'A name', lambda mmsi: real_values.get(mmsi))
            objective.prediction = prediction

            loss, _ = objective.build_trainer(None, mmsis)

            self.assertAlmostEqual(1.0, loss.eval())


class FishingLocalisationLossTest(tf.test.TestCase):
    def test_simple_loss(self):
        with self.test_session():
            logits = np.array([[1, 0, 0, 1, 0], [1, 0, 1, 1, 0]], np.float32)
            targets = np.array([[1, 0, -1, 0, -1], [1, 0, 0, -1, -1]],
                               np.float32)

            loss = utility.fishing_localisation_loss(logits, targets)

            filtered_logits = logits[targets != -1]
            filtered_targets = targets[targets != -1]

            filtered_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits,
                                                        labels=filtered_targets))

            self.assertAlmostEqual(filtered_loss.eval(), loss.eval(), places=5)

    def test_loss_scaling_floor(self):
        with self.test_session():
            logits = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32)
            targets = np.array([[0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
                               np.float32)

            loss = utility.fishing_localisation_loss(logits, targets)

            self.assertAlmostEqual(0.66164052, loss.eval())

    def test_loss_no_targets(self):
        with self.test_session():
            logits = np.array([[0, 0, 0, 0, 0]], np.float32)
            targets = np.array([[-1, -1, -1, -1, -1]], np.float32)

            loss = utility.fishing_localisation_loss(logits, targets)

            self.assertAlmostEqual(0.0, loss.eval())


# Check we are actually getting vessels with fishing
# localisation info (check loading the metadata, and choosing the
# segments).
class ObjectiveFunctionsTest(tf.test.TestCase):
    vmd_dict = {'Test': {100001: ({'label': 'Longliner'}, 1.0)}}
    range1 = utility.FishingRange(
        _dt("2015-04-01T06:00:00Z"), _dt("2015-04-01T09:30:0Z"), 1.0)
    range2 = utility.FishingRange(
        _dt("2015-04-01T09:30:00Z"), _dt("2015-04-01T12:30:01Z"), 0.0)

    def _build_trainer(self, logits, objective):
        timestamps = [
            _dt("2015-04-01T08:30:00Z"),
            _dt("2015-04-01T09:00:00Z"),
            _dt("2015-04-01T09:29:00Z"),
            _dt("2015-04-01T10:00:00Z"),
            _dt("2015-04-01T10:30:00Z"),
            _dt("2015-04-01T11:00:00Z"),
        ]
        epoch_timestamps = [[calendar.timegm(t.utctimetuple())
                             for t in timestamps]]
        mmsis = [100001]

        objective.build(logits)
        return objective.build_trainer(epoch_timestamps, mmsis)


if __name__ == '__main__':
    tf.test.main()
