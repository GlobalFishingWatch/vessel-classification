from __future__ import absolute_import
import argparse
import json
import logging
import math
import os
from pkg_resources import resource_filename
import sys
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
        with tf.Graph().as_default():
            if config.is_worker():
                # This task is a worker, running training and sharing parameters with
                # other workers via the parameter server.
                with tf.device(
                        tf.train.replica_device_setter(
                            worker_device='/job:%s/task:%d' % (
                                config.task_type, config.task_index),
                            cluster=config.cluster_spec)):
                    # The chief worker is responsible for writing out checkpoints as the
                    # run progresses.
                    trainer.run_training(server.target, config.is_chief())
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
        resource_filename('classification.data', 'net_training_20161016.csv'))
    if not os.path.exists(metadata_file):
        logging.fatal("Could not find metadata file: %s.", metadata_file)
        sys.exit(-1)

    fishing_range_file = os.path.abspath(
        resource_filename('classification.data',
                          'combined_fishing_ranges.csv'))
    if not os.path.exists(fishing_range_file):
        logging.fatal("Could not find fishing range file: %s.",
                      fishing_range_file)
        sys.exit(-1)

    fishing_ranges = utility.read_fishing_ranges(fishing_range_file)

    all_available_mmsis = utility.find_available_mmsis(args.root_feature_path)

    vessel_metadata = utility.read_vessel_multiclass_metadata(
        all_available_mmsis, metadata_file)

    coarse_label_objective = utility.ClassificationObjective('Vessel class', 'label', utility.VESSEL_CLASS_NAMES)
    fine_label_objective = utility.ClassificationObjective('Vessel detailed class', 'sublabel', utility.VESSEL_CLASS_DETAILED_NAMES)
    training_objectives = [
        coarse_label_objective,
        fine_label_objective
    ]
    feature_dimensions = int(args.feature_dimensions)
    model = Model(feature_dimensions, training_objectives)
    
    trainer = Trainer(model, vessel_metadata, fishing_ranges,
                      args.root_feature_path, args.training_output_path)

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

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
