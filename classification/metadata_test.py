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
import numpy as np
from . import metadata
import tensorflow as tf
from datetime import datetime
import six


class VesselMetadataFileReaderTest(tf.test.TestCase):
    raw_lines = [
        'id,label,length,split,idhash\n',
        '100001,drifting_longlines,10.0,Test,2\n',
        '100002,drifting_longlines,24.0,Training,3\n',
        '100003,drifting_longlines,7.0,Training,4\n',
        '100004,drifting_longlines,8.0,Test,5\n',
        '100005,trawlers,10.0,Test,6\n',
        '100006,trawlers,24.0,Test,7\n',
        '100007,passenger,24.0,Training,8\n',
        '100008,trawlers,24.0,Training,9\n',
        '100009,trawlers,10.0,Test,10\n',
        '100010,trawlers,24.0,Training,11\n',
        '100011,tug,60.0,Test,12\n',
        '100012,tug,5.0,Training,13\n',
        '100014,tug,24.0,Test,14\n',
        '100013,tug|trawlers,5.0,Training,15\n',
    ]

    fishing_range_dict = {
        b'100001': [metadata.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        b'100002': [metadata.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        b'100003': [metadata.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        b'100004': [metadata.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        b'100005': [metadata.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        b'100006': [metadata.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        b'100007': [metadata.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        b'100008': [metadata.FishingRange(
            datetime(2015, 3, 1), datetime(2015, 3, 2), 1.0)],
        b'100009':
        [metadata.FishingRange(datetime(2015, 3, 1), datetime(2015, 3, 4), 1.0)
         ],  # Thrice as much fishing
        b'100010': [],
        b'100011': [],
        b'100012': [],
        b'100013': [],
    }

    def test_metadata_file_reader(self):
        parsed_lines = csv.DictReader(self.raw_lines)
        available_vessels = set(six.ensure_binary(str(x)) for x in range(100001, 100014))
        result = metadata.read_vessel_multiclass_metadata_lines(
            available_vessels, parsed_lines, {})

        # First one is test so weighted as 1 for now
        self.assertEqual(1.0, result.vessel_weight(b'100001'))
        self.assertEqual(1.118033988749895, result.vessel_weight(b'100002'))
        self.assertEqual(1.0, result.vessel_weight(b'100008'))
        self.assertEqual(1.2909944487358056, result.vessel_weight(b'100012'))
        self.assertEqual(1.5811388300841898, result.vessel_weight(b'100007'))
        self.assertEqual(1.1454972243679027, result.vessel_weight(b'100013'))

        self._check_splits(result)

    def test_fixed_time_reader(self):
        parsed_lines = csv.DictReader(self.raw_lines)
        available_vessels = set(six.ensure_binary(str(x)) for x in range(100001, 100014))
        result = metadata.read_vessel_time_weighted_metadata_lines(
            available_vessels, parsed_lines, self.fishing_range_dict,
            'Test')

        self.assertEqual(1.0, result.vessel_weight(b'100001'))
        self.assertEqual(1.0, result.vessel_weight(b'100002'))
        self.assertEqual(3.0, result.vessel_weight(b'100009'))
        self.assertEqual(0.0, result.vessel_weight(b'100012'))

        self._check_splits(result)

    def _check_splits(self, result):

        self.assertTrue('Training' in result.metadata_by_split)
        self.assertTrue('Test' in result.metadata_by_split)
        self.assertTrue('passenger', result.vessel_label('label', b'100007'))

        print(result.metadata_by_split['Test'][b'100001'][0])
        self.assertEqual(result.metadata_by_split['Test'][b'100001'][0],
                          {'label': 'drifting_longlines',
                           'length': '10.0',
                           'id': '100001',
                           'split': 'Test',
                           'idhash' : '2'})
        self.assertEqual(result.metadata_by_split['Test'][b'100005'][0],
                          {'label': 'trawlers',
                           'length': '10.0',
                           'id': '100005',
                           'split': 'Test',
                           'idhash' : '6'})
        self.assertEqual(result.metadata_by_split['Training'][b'100002'][0],
                          {'label': 'drifting_longlines',
                           'length': '24.0',
                           'id': '100002',
                           'split': 'Training',
                           'idhash' : '3'})
        self.assertEqual(result.metadata_by_split['Training'][b'100003'][0],
                          {'label': 'drifting_longlines',
                           'length': '7.0',
                           'id': '100003',
                           'split': 'Training',
                           'idhash' : '4'})


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
            for row in metadata.metadata_file_reader(metadata_file):
                label_str = row['label']
                for lbl in label_str.split('|'):
                    labels.add(lbl.strip())
            labels.remove('')

            expected = set([lbl for (lbl, _) in metadata.VESSEL_CATEGORIES])
            assert expected >= labels, (expected - labels, labels - expected)


class MultihotLabelConsistencyTest(tf.test.TestCase):
    def test_fine_label_consistency(self):
        names = []
        for coarse, fine_list in metadata.VESSEL_CATEGORIES:
            for fine in fine_list:
                if fine not in names:
                    names.append(fine)
        self.assertEqual(
            sorted(names), sorted(metadata.VESSEL_CLASS_DETAILED_NAMES))


if __name__ == '__main__':
    tf.test.main()
