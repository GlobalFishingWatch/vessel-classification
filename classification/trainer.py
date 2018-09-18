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
import argparse
import json
import logging
import math
import numpy as np
import os
import sys
from . import utility
from . import feature_generation

import tensorflow as tf

# TODO: ==> just a function in run_training.
class Trainer:
    """ Handles the mechanics of training and evaluating a vessel behaviour
        model.
    """

    num_parallel_readers = 32

    def __init__(self, model, base_feature_path, checkpoint_dir):
        self.model = model
        self.base_feature_path = base_feature_path
        self.checkpoint_dir = checkpoint_dir

    def run_training(self):
        """ The function for running a training replica on a worker. """
        train_input_fn = self.model.make_training_input_fn(self.base_feature_path, self.num_parallel_readers)
        test_input_fn = self.model.make_test_input_fn(self.base_feature_path, self.num_parallel_readers)
        estimator = self.model.make_estimator(self.checkpoint_dir)
        train_spec = tf.estimator.TrainSpec(
                        input_fn=train_input_fn, 
                        max_steps=self.model.number_of_steps)
        eval_spec = tf.estimator.EvalSpec(
                        input_fn=test_input_fn,
                        start_delay_secs=120,
                        throttle_secs=300
                        )

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

