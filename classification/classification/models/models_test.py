import numpy as np
import tensorflow as tf

from classification import utility
from alex import vessel_classification, vessel_and_fishing_range_classification
from tim import mixed_classification_1, mixed_classification_multi_1

# TODO(alexwilson): Feed some data in. Also check evaluation.build_json_results


class ModelsTest(tf.test.TestCase):
    num_feature_dimensions = 11
    model_classes = [mixed_classification_1.Model, mixed_classification_multi_1.Model, 
                     vessel_classification.Model, vessel_and_fishing_range_classification.Model]

    def _build_model_input(self, model):
        feature = [0.0] * model.num_feature_dimensions
        features = np.array([[[feature] * model.window_max_points]] *
                            model.batch_size, np.float32)
        timestamps = np.array([[0] * model.window_max_points] *
                              model.batch_size, np.int32)
        mmsis = np.array([0] * model.batch_size, np.int32)

        return tf.constant(features), tf.constant(timestamps), tf.constant(
            mmsis)

    def _build_model_training_net(self, model_class):
        vmd = utility.VesselMetadata({}, {})
        model = model_class(self.num_feature_dimensions, vmd)
        features, timestamps, mmsis = self._build_model_input(model)

        return model.build_training_net(features, timestamps, mmsis)

    def _build_model_inference_net(self, model_class):
        vmd = utility.VesselMetadata({}, {})
        model = model_class(self.num_feature_dimensions, vmd)
        features, timestamps, mmsis = self._build_model_input(model)

        return model.build_inference_net(features, timestamps, mmsis)

    def test_model_training_nets(self):
        for i, model_class in enumerate(self.model_classes):
            with self.test_session():
                with tf.variable_scope("training-test-{}".format(i)):
                    optimizer, trainers = self._build_model_training_net(
                        model_class)

    def test_model_inference_nets(self):
        for i, model_class in enumerate(self.model_classes):
            with self.test_session():
                with tf.variable_scope("inference-test-{}".format(i)):
                    evaluations = self._build_model_inference_net(model_class)

                    for e in evaluations:
                        e.build_test_metrics()


if __name__ == '__main__':
    tf.test.main()
