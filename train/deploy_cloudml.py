#!/usr/bin/env python

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

from __future__ import print_function
from common.gcp_config import GcpConfig
import yaml
import json
import time
import subprocess
import os
import datetime
import argparse
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import tempfile


def launch(args):
    # Read the configuration file so that we 
    # know the train path and don't need to
    # hardcode it here
    with open(args.config_file) as f:
        config = yaml.safe_load(f.read())
        tf_config_template = config['tensor_flow_config_template']

    gcp = GcpConfig.make_from_env_name(args.env, args.job_name)

    tf_config_txt = tf_config_template.format(
        output_path=gcp.model_path(), **args.__dict__)

    timestamp = gcp.start_time.strftime('%Y%m%dT%H%M%S')
    job_id = ('%s_%s_%s' % (args.model_name, args.job_name, timestamp)).replace(
        '.', '_').replace('-', '_')

    # Kick off the job on CloudML
    with tempfile.NamedTemporaryFile('w') as temp:
        temp.write(tf_config_txt)
        temp.flush()

        with open(temp.name) as f:
            tf_config = yaml.safe_load(f)

        # It seems that we currently need to pass args as both 'args' in the
        # config file and as args after the '--'?!
        args = [
            'gcloud', 'ai-platform', 
            'jobs', 'submit', 'training', job_id,
            '--config', temp.name, '--module-name',
            'classification.run_training', '--staging-bucket',
            config['staging_bucket'], '--package-path', 'classification',
            '--region', config['region'], '--'
        ] + tf_config['trainingInput']['args']

        print('Executing:\n', ' '.join(args))
        print("Config:\n", tf_config_txt)

        subprocess.check_call(args)

    return job_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Deploy ML Training.')
    parser.add_argument('--env', required=True,
                        help='environment for run: prod/dev.')
    parser.add_argument('--model_name', required=True,
                        help='module name of model.')
    parser.add_argument('--job_name', required=True,
                        help='unique name for this job.')
    parser.add_argument('--feature_path', required=True,
                        help='gcs path to features.') 
    parser.add_argument('--vessel_info', required=True,
                        help='local path to vessel_info.')
    parser.add_argument('--fishing_ranges', default='',
                        help='optional local path fishing ranges')
    parser.add_argument('--config_file', default='deploy_cloudml.yaml',
                        help='configuration file path.')
    parser.add_argument('--split', default=0, type=int,
                        help='Split to use (-1) for all')
    args = parser.parse_args()

    launch(args)
