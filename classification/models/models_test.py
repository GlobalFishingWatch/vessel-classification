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

from classification import metadata
from . import vessel_characterization, fishing_detection 


class ModelsTest(tf.test.TestCase):
    num_feature_dimensions = 11
    model_classes = [vessel_characterization.Model, fishing_detection.Model]

    def _build_estimator(self, model_class):
        vmd = metadata.VesselMetadata({}, {})
        model = model_class(self.num_feature_dimensions, vmd, metrics='all')
        return model.make_estimator("dummy_directory")

    def test_estimator_contruction(self):
        for i, model_class in enumerate(self.model_classes):
            with self.test_session():
                # This protects against multiple model using same variable names
                with tf.variable_scope("training-test-{}".format(i)):
                    est = self._build_estimator(model_class)

    # TODO: test input_fn


if __name__ == '__main__':
    tf.test.main()
