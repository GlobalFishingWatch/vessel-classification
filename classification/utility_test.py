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

import os
import csv
import model
import numpy as np
import utility
import tensorflow as tf
from datetime import datetime


class PythonReshapePadTest(tf.test.TestCase):
    def testRepeatTensor(self):
        with self.test_session():
            batch_size = 1
            depth = 3
            width = 4
            input_data = tf.constant(
                np.array([[[[1., 2.], [4., 5.], [7., 8.]]]]))

            double_padded = utility.repeat_tensor(input_data, 3)

            expected = np.array([[[[1., 2.], [1., 2.], [1., 2.], [4., 5.],
                                   [4., 5.], [4., 5.], [7., 8.], [7., 8.],
                                   [7., 8.]]]])

            self.assertAllEqual(double_padded.eval(), expected)


class _FakeRandint(object):
    def randint(start, stop=0, step=0):
        return 0


class PythonFixedTimeExtractTest(tf.test.TestCase):
    def test_cropped_extract(self):
        with self.test_session():
            input_data = np.array([[1., 5.], [2., 4.], [3., 7.], [4., 9.],
                                   [5., 3.], [6., 8.], [7., 2.], [8., 9.]])

            expected_result = np.array([[1., 5.], [2., 4.], [3., 7.], [4., 9.],
                                        [5., 3.], [6., 8.], [1., 5.],
                                        [2., 4.]])

            class FakeRandomState(object):
                def randint(self, min, max):
                    return 0

            res = utility.np_array_random_fixed_time_extract(
                FakeRandomState(), input_data, 5, 8, 50)

            self.assertAllEqual(res, expected_result)

    def test_uncropped_extract(self):
        with self.test_session():
            input_data = np.array([[1., 5., 6.], [2., 4., 4.], [3., 7., 9.],
                                   [4., 9., 0.]])

            res = utility.np_array_random_fixed_time_extract(
                _FakeRandint(), input_data, 20, 4, 50)
            self.assertAllEqual(res, input_data)

    def test_uncropped_extract_pad(self):
        with self.test_session():
            input_data = np.array([[1., 5., 6.], [2., 4., 4.], [3., 7., 9.]])
            expected_result = np.array([[1., 5., 6.], [2., 4., 4.],
                                        [3., 7., 9.], [1., 5.,
                                                       6.], [2., 4., 4.]])

            res = utility.np_array_random_fixed_time_extract(
                _FakeRandint(), input_data, 20, 5, 50)
            self.assertAllEqual(res, expected_result)


class VesselMetadataFileReaderTest(tf.test.TestCase):
    raw_lines = [
        'mmsi,label,length,split\n',
        '100001,Longliner,10.0,Test\n',
        '100002,Longliner,24.0,Training\n',
        '100003,Longliner,7.0,Training\n',
        '100004,Longliner,8.0,Test\n',
        '100005,Trawler,10.0,Test\n',
        '100006,Trawler,24.0,Test\n',
        '100007,Passenger,24.0,Training\n',
        '100008,Trawler,24.0,Training\n',
        '100009,Trawler,10.0,Test\n',
        '100010,Trawler,24.0,Training\n',
        '100011,Tug,60.0,Test\n',
        '100012,Tug,5.0,Training\n',
        '100013,Tug,24.0,Test\n',
    ]

    fishing_range_dict = {
        '100001': [utility.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        '100002': [utility.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        '100003': [utility.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        '100004': [utility.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        '100005': [utility.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        '100006': [utility.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        '100007': [utility.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        '100008': [utility.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        '100009':
        [utility.FishingRange(datetime(2015, 3, 1), datetime(2015, 3, 4), 1.0)
         ],  # Thrice as much fishing
        '100010': [],
        '100011': [],
        '100012': [],
        '100013': [],
    }

    def test_metadata_file_reader(self):
        parsed_lines = csv.DictReader(self.raw_lines)
        available_vessels = set(str(x) for x in range(100001, 100013))
        result = utility.read_vessel_multiclass_metadata_lines(
            available_vessels, parsed_lines, {}, 1)

        self.assertEquals(1.2247448713915889, result.vessel_weight('100001'))
        self.assertEquals(1.0, result.vessel_weight('100002'))
        self.assertEquals(1.7320508075688772, result.vessel_weight('100011'))

        self._check_splits(result)

    def test_fixed_time_reader(self):
        parsed_lines = csv.DictReader(self.raw_lines)
        available_vessels = set(str(x) for x in range(100001, 100013))
        result = utility.read_vessel_time_weighted_metadata_lines(
            available_vessels, parsed_lines, self.fishing_range_dict)

        self.assertEquals(1.0, result.vessel_weight('100001'))
        self.assertEquals(1.0, result.vessel_weight('100002'))
        self.assertEquals(3.0, result.vessel_weight('100009'))
        self.assertEquals(0.0, result.vessel_weight('100012'))

        self._check_splits(result)

    def _check_splits(self, result):

        self.assertTrue('Training' in result.metadata_by_split)
        self.assertTrue('Test' in result.metadata_by_split)
        self.assertTrue('Passenger', result.vessel_label('label', '100007'))

        self.assertEquals(result.metadata_by_split['Test']['100001'][0],
                          {'label': 'Longliner',
                           'length': '10.0',
                           'mmsi': '100001',
                           'split': 'Test'})
        self.assertEquals(result.metadata_by_split['Test']['100005'][0],
                          {'label': 'Trawler',
                           'length': '10.0',
                           'mmsi': '100005',
                           'split': 'Test'})
        self.assertEquals(result.metadata_by_split['Training']['100002'][0],
                          {'label': 'Longliner',
                           'length': '24.0',
                           'mmsi': '100002',
                           'split': 'Training'})
        self.assertEquals(result.metadata_by_split['Training']['100003'][0],
                          {'label': 'Longliner',
                           'length': '7.0',
                           'mmsi': '100003',
                           'split': 'Training'})


def _get_metadata_files():
    from pkg_resources import resource_filename
    for name in ["training_classes.csv"]:
        # TODO: rework to test encounters as well.
        yield os.path.abspath(resource_filename('classification.data', name))


class MetadataConsistencyTest(tf.test.TestCase):
    def test_metadata_consistency(self):
        for metadata_file in _get_metadata_files():
            self.assertTrue(os.path.exists(metadata_file))
            # By putting '' in these sets we can safely remove it later
            labels = set([''])
            for row in utility.metadata_file_reader(metadata_file):
                label_str = row['label']
                for lbl in label_str.split('|'):
                    labels.add(lbl.strip())
            labels.remove('')

            expected = set([lbl for (lbl, _) in utility.VESSEL_CATEGORIES])
            assert expected >= labels, (expected - labels, labels - expected)


class MultihotLabelConsistencyTest(tf.test.TestCase):
    def test_fine_label_consistency(self):
        names = []
        for coarse, fine_list in utility.VESSEL_CATEGORIES:
            for fine in fine_list:
                if fine not in names:
                    names.append(fine)
        self.assertEquals(
            sorted(names), sorted(utility.VESSEL_CLASS_DETAILED_NAMES))


if __name__ == '__main__':
    tf.test.main()
