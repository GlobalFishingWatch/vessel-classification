from __future__ import absolute_import
import argparse
import json
from . import layers
import logging
import math
import os
from  . import utility
import importlib

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as metrics



def run_training(config, server, trainer):
  logging.info("Starting training task %s, %d", config.task_type, config.task_index)
  if config.is_ps():
    # This task is a parameter server.
    server.join()
  else:
    with tf.Graph().as_default():
      if config.is_worker():
        # This task is a worker, running training and sharing parameters with
        # other workers via the parameter server.
        with tf.device(tf.train.replica_device_setter(
            worker_device = '/job:%s/task:%d' % (config.task_type, config.task_index),
            cluster = config.cluster_spec)):
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
    Trainer = importlib.import_module(module).Trainer
  except:
    logging.error("Could not load model: {}".format(module))
    raise

  trainer = Trainer(args.root_feature_path, args.training_output_path)

  config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  if (config == {}):
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

  argparser.add_argument('--root_feature_path', required=True,
      help='The root path to the vessel movement feature directories.')

  argparser.add_argument('--training_output_path', required=True,
      help='The working path for model statistics and checkpoints.')

  return argparser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  main(args)
