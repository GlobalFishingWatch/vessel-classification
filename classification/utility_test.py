from models.alex import layers
import numpy as np
import utility
import tensorflow as tf


class InceptionLayerTest(tf.test.TestCase):
    def test_layer_shape(self):
        with self.test_session():
            input_data = [[1., 5., 6.], [2., 4., 4.], [3., 7., 9.],
                          [4., 9., 0.], [3., 7., 9.], [4., 9., 0.]]
            # Add an outer dimension to take the data from 1d to 2d
            input_data = tf.expand_dims(input_data, 0)
            # Add an outer dimension to take the data from unbatched to batch
            input_data = tf.expand_dims(input_data, 0)
            input_data_shape = tf.shape(input_data)
            self.assertAllEqual(input_data_shape.eval(), [1, 1, 6, 3])

            res = layers.misconception_layer(input_data, 3, 2, 5, True)
            res_shape = tf.shape(res)

            tf.initialize_all_variables().run()

            self.assertAllEqual(res_shape.eval(), [1, 1, 3, 5])


class PythonFixedTimeExtractTest(tf.test.TestCase):
    def test_cropped_extract(self):
        with self.test_session():
            input_data = np.array([[1., 5.], [2., 4.], [3., 7.], [4., 9.],
                                   [5., 3.], [6., 8.], [7., 2.], [8., 9.]])
            expected_result = np.array([[1., 5.], [2., 4.], [3., 7.], [4., 9.],
                                        [5.,
                                         3.], [6., 8.], [1., 5.], [2., 4.]])

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


class VesselMetadataFileReader(tf.test.TestCase):
    def test_metadata_file_reader(self):
        lines = [
            'mmsi,split,kind\n',
            '100001,Training,Longliner\n',
            '100002,Training,Longliner\n',
            '100003,Training,Longliner\n',
            '100004,Training,Longliner\n',
            '100005,Training,Trawler\n',
            '100006,Training,Trawler\n',
            '100007,Training,Passenger\n',
            '100008,Test,Trawler\n',
            '100009,Test,Trawler\n',
            '100010,Test,Trawler\n',
            '100011,Test,Tug\n',
            '100012,Test,Tug\n',
        ]
        result = utility.read_vessel_metadata_file_lines(lines)

        self.assertTrue('Training' in result)
        self.assertEquals(result['Training'][100001], ('Longliner', 1.0))
        self.assertEquals(result['Training'][100002], ('Longliner', 1.0))
        self.assertEquals(result['Training'][100003], ('Longliner', 1.0))
        self.assertEquals(result['Training'][100004], ('Longliner', 1.0))
        self.assertEquals(result['Training'][100005], ('Trawler', 2.0))
        self.assertEquals(result['Training'][100006], ('Trawler', 2.0))
        self.assertEquals(result['Training'][100007], ('Passenger', 4.0))

        self.assertTrue('Test' in result)
        self.assertEquals(result['Test'][100008], ('Trawler', 1.0))
        self.assertEquals(result['Test'][100009], ('Trawler', 1.0))
        self.assertEquals(result['Test'][100010], ('Trawler', 1.0))
        self.assertEquals(result['Test'][100011], ('Tug', 3.0 / 2.0))
        self.assertEquals(result['Test'][100012], ('Tug', 3.0 / 2.0))


if __name__ == '__main__':
    tf.test.main()
