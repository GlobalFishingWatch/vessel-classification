import tensorflow as tf
import numpy as np
import utility


class HelperFunctionsTesT(tf.test.TestCase):
  def test_extract_features(self):
    input_data = np.array([[1,2,3,4,5,6,7], [8,9,10,11,12,13,14]], dtype=np.float32)
    expected_data = np.array([[np.log(2),np.log(3),np.log(4),np.log(5),6,np.log(7)],
      [np.log(9),np.log(10),np.log(11),np.log(12),13,np.log(14)]], dtype=np.float32)

    res = utility.extract_features(input_data, 100, 2)
    self.assertAllEqual(res, expected_data)

class InceptionLayerTest(tf.test.TestCase):
  def test_layer_shape(self):
    with self.test_session():
      input_data = [[1., 5., 6.], [2., 4., 4.], [3., 7., 9.], [4., 9., 0.],
          [3., 7., 9.], [4., 9., 0.]]
      # Add an outer dimension to take the data from 1d to 2d
      input_data = tf.expand_dims(input_data, 0)
      # Add an outer dimension to take the data from unbatched to batch
      input_data = tf.expand_dims(input_data, 0)
      input_data_shape = tf.shape(input_data)
      self.assertAllEqual(input_data_shape.eval(), [1, 1, 6, 3])

      res = utility.inception_layer(input_data, 3, 2, 5)
      res_shape = tf.shape(res)

      tf.initialize_all_variables().run()

      self.assertAllEqual(res_shape.eval(), [1, 1, 3, 10])

class PythonFixedTimeExtractTest(tf.test.TestCase):
  def test_cropped_extract(self):
    with self.test_session():
      input_data = np.array([[1., 5.], [2., 4.], [3., 7.], [4., 9.], [5., 3.],
        [6., 8.], [7., 2.], [8., 9.]])
      expected_result = np.array([[1., 5.], [2., 4.], [3., 7.], [4., 9.], [5.,
        3.], [6., 8.], [1., 5.], [2., 4.]])

      res = utility.np_array_fixed_time_extract(input_data, 5, 8)
      
      self.assertAllEqual(res, expected_result)

  def test_uncropped_extract(self):
    with self.test_session():
      input_data = np.array([[1., 5., 6.], [2., 4., 4.], [3., 7., 9.], [4., 9.,
        0.]])

      res = utility.np_array_fixed_time_extract(input_data, 20, 4)
      self.assertAllEqual(res, input_data)

  def test_uncropped_extract_pad(self):
    with self.test_session():
      input_data = np.array([[1., 5., 6.], [2., 4., 4.], [3., 7., 9.]])
      expected_result = np.array([[1., 5., 6.], [2., 4., 4.], [3., 7., 9.], [1.,
        5., 6.], [2., 4., 4.]])

      res = utility.np_array_fixed_time_extract(input_data, 20, 5)
      self.assertAllEqual(res, expected_result)

if __name__ == '__main__':
  tf.test.main()
