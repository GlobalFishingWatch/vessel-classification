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

import numpy as np
import tensorflow as tf

from classification import utility
from prod import vessel_characterization, fishing_detection as fishing_detection

# TODO(alexwilson): Feed some data in. Also check evaluation.build_json_results


class ModelsTest(tf.test.TestCase):
    num_feature_dimensions = 11
    model_classes = [vessel_characterization.Model, fishing_detection.Model]

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
        model = model_class(self.num_feature_dimensions, vmd, metrics='all')
        features, timestamps, mmsis = self._build_model_input(model)

        return model.build_training_net(features, timestamps, mmsis)

    def _build_model_inference_net(self, model_class):
        vmd = utility.VesselMetadata({}, {})
        model = model_class(self.num_feature_dimensions, vmd, metrics='all')
        features, timestamps, mmsis = self._build_model_input(model)

        return model.build_inference_net(features, timestamps, mmsis)

    def test_model_training_nets(self):
        for i, model_class in enumerate(self.model_classes):
            with self.test_session():
                # This protects against multiple model using same variable names
                with tf.variable_scope("training-test-{}".format(i)):
                    optimizer, trainers = self._build_model_training_net(
                        model_class)

    def test_model_inference_nets(self):
        for i, model_class in enumerate(self.model_classes):
            with self.test_session():
                # This protects against multiple model using same variable names
                with tf.variable_scope("inference-test-{}".format(i)):
                    evaluations = self._build_model_inference_net(model_class)

                    for e in evaluations:
                        e.build_test_metrics()


if __name__ == '__main__':
    tf.test.main()
