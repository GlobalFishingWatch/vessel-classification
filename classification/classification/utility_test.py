import csv
import os
import model
import numpy as np
import utility
import tensorflow as tf


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
                lambda _: 0, input_data, 20, 4, 50)
            self.assertAllEqual(res, input_data)

    def test_uncropped_extract_pad(self):
        with self.test_session():
            input_data = np.array([[1., 5., 6.], [2., 4., 4.], [3., 7., 9.]])
            expected_result = np.array([[1., 5., 6.], [2., 4., 4.],
                                        [3., 7., 9.], [1., 5.,
                                                       6.], [2., 4., 4.]])

            res = utility.np_array_random_fixed_time_extract(
                lambda _: 0, input_data, 20, 5, 50)
            self.assertAllEqual(res, expected_result)


class VesselMetadataFileReaderTest(tf.test.TestCase):
    def test_metadata_file_reader(self):
        raw_lines = [
            'mmsi,label,length\n',
            '100001,Longliner,10.0\n',
            '100002,Longliner,24.0\n',
            '100003,Longliner,7.0\n',
            '100004,Longliner,8.0\n',
            '100005,Trawler,10.0\n',
            '100006,Trawler,24.0\n',
            '100007,Passenger,24.0\n',
            '100008,Trawler,24.0\n',
            '100009,Trawler,10.0\n',
            '100010,Trawler,24.0\n',
            '100011,Tug,60.0\n',
            '100012,Tug,5.0\n',
            '100013,Tug,24.0\n',
        ]
        parsed_lines = csv.DictReader(raw_lines)
        available_vessels = set(range(100001, 100013))
        result = utility.read_vessel_multiclass_metadata_lines(
            available_vessels, parsed_lines, {}, 1)

        self.assertEquals(3.0, result.vessel_weight(100001))
        self.assertEquals(1.0, result.vessel_weight(100002))
        self.assertEquals(1.5, result.vessel_weight(100011))

        self.assertTrue('Training' in result.metadata_by_split)
        self.assertTrue('Test' in result.metadata_by_split)
        self.assertTrue('Passenger', result.vessel_label('label', 100007))

        self.assertEquals(result.metadata_by_split['Test'][100001],
                          ({'label': 'Longliner',
                            'length': '10.0',
                            'mmsi': '100001'}, 3.0))
        self.assertEquals(result.metadata_by_split['Test'][100005],
                          ({'label': 'Trawler',
                            'length': '10.0',
                            'mmsi': '100005'}, 1.0))

        self.assertEquals(result.metadata_by_split['Training'][100002],
                          ({'label': 'Longliner',
                            'length': '24.0',
                            'mmsi': '100002'}, 1.0))
        self.assertEquals(result.metadata_by_split['Training'][100003],
                          ({'label': 'Longliner',
                            'length': '7.0',
                            'mmsi': '100003'}, 1.0))


class MetadataConsistencyTest(tf.test.TestCase):
    def test_metadata_consistency(self):
        from pkg_resources import resource_filename
        metadata_file = os.path.abspath(
            resource_filename('data', 'net_training_20161115.csv'))
        self.assertTrue(os.path.exists(metadata_file))

        is_fishing_labels = set()
        coarse_labels = set([''])   # By putting '' in these sets we can safely remove it later
        fine_labels = set([''])
        for row in utility.metadata_file_reader(metadata_file):
            is_fishing_labels.add(row['is_fishing'].strip())
            coarse_labels.add(row['label'].strip())
            fine_labels.add(row['sublabel'].strip())
        coarse_labels.remove('')
        fine_labels.remove('')

        # Is fishing should never be blank
        self.assertFalse('' in is_fishing_labels)

        self.assertEquals(is_fishing_labels, set(utility.FISHING_NONFISHING_NAMES))

        self.assertEquals(coarse_labels, set(utility.VESSEL_CLASS_NAMES))

        self.assertEquals(fine_labels, set(utility.VESSEL_CLASS_DETAILED_NAMES))


if __name__ == '__main__':
    tf.test.main()
