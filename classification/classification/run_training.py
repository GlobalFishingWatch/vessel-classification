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
import os
from pkg_resources import resource_filename
import sys
from . import model
from . import utility
from .trainer import Trainer
import importlib
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics


def run_training(config, server, trainer):
    logging.info("Starting training task %s, %d", config.task_type,
                 config.task_index)
    if config.is_ps():
        # This task is a parameter server.
        server.join()
    else:
        if config.is_worker():
            # This task is a worker, running training and sharing parameters with
            # other workers via the parameter server.
            device = tf.train.replica_device_setter(
                worker_device='/job:%s/task:%d' % (config.task_type,
                                                   config.task_index),
                cluster=config.cluster_spec)
            # The chief worker is responsible for writing out checkpoints as the
            # run progresses.
            trainer.run_training(
                server.target, config.is_chief(), device=device)
        elif config.is_master():
            # This task is the master, running evaluation.
            trainer.run_evaluation(server.target)
        else:
            raise ValueError('Unexpected task type: %s', config.task_type)


def main(args):
    logging.getLogger().setLevel(logging.DEBUG)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    logging.info("Running with Tensorflow version: %s", tf.__version__)

    logging.info("Loading model: %s", args.model_name)

    module = "classification.models.{}".format(args.model_name)
    try:
        Model = importlib.import_module(module).Model
    except:
        logging.fatal("Could not load model: {}".format(module))
        raise

    metadata_file = os.path.abspath(
        resource_filename('classification.data', args.metadata_file))
    if not os.path.exists(metadata_file):
        logging.fatal("Could not find metadata file: %s.", metadata_file)
        sys.exit(-1)

    fishing_range_file = os.path.abspath(
        resource_filename('classification.data', args.fishing_ranges_file))
    if not os.path.exists(fishing_range_file):
        logging.fatal("Could not find fishing range file: %s.",
                      fishing_range_file)
        sys.exit(-1)

    fishing_ranges = utility.read_fishing_ranges(fishing_range_file)

    all_available_mmsis = utility.find_available_mmsis(args.root_feature_path)

    vessel_metadata = Model.read_metadata(
        all_available_mmsis, metadata_file,
        fishing_ranges, int(args.fishing_range_training_upweight))

    feature_dimensions = int(args.feature_dimensions)
    chosen_model = Model(feature_dimensions, vessel_metadata, args.metrics)

    # TODO: training verbosity --training-verbosity
    trainer = Trainer(chosen_model, args.root_feature_path,
                      args.training_output_path)

    config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    if (config == {}):
        logging.info("Running locally, training only...")
        node_config = utility.ClusterNodeConfig.create_local_server_config()
        server = tf.train.Server.create_local_server()
        run_training(node_config, server, trainer)
    else:
        logging.info("Config dictionary: %s", config)

        node_config = utility.ClusterNodeConfig(config)
        server = node_config.create_server()

        run_training(node_config, server, trainer)


def parse_args():
    """ Parses command-line arguments for training."""
    argparser = argparse.ArgumentParser('Train fishing classification model.')

    argparser.add_argument('model_name')

    argparser.add_argument(
        '--root_feature_path',
        required=True,
        help='The root path to the vessel movement feature directories.')

    argparser.add_argument(
        '--training_output_path',
        required=True,
        help='The working path for model statistics and checkpoints.')

    argparser.add_argument(
        '--feature_dimensions',
        required=True,
        help='The number of dimensions of a classification feature.')

    argparser.add_argument(
        '--fishing_range_training_upweight',
        default=1.0,
        help='The amount to upweight vessels that have fishing ranges when training.')

    argparser.add_argument('--metadata_file', help='Path to metadata.')

    argparser.add_argument(
        '--fishing_ranges_file', help='Path to fishing range file.')

    argparser.add_argument(
        '--metrics',
        default='all',
        help='How many metrics to dump ["all" | "minimal"]')

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
