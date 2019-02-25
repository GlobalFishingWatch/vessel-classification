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

import datetime
import logging
import os
import sys


class GcpConfig(object):
    def __init__(self, start_time, project_id, root_path):
        self.start_time = start_time
        self.project_id = project_id
        self.root_path = root_path

    def model_path(self):
        return self.root_path + '/models'

    # TODO(alexwilson): This config is too hard-coded to our current setup. Move
    # out to config files for greater flexibility. Note there is an equivalent to
    # this in Commmon.scala which should remain in-sync.
    @staticmethod
    def make_from_env_name(environment, job_id):
        now = datetime.datetime.utcnow()
        project_id = "world-fishing-827"
        if environment == 'prod':
            root_path = 'gs://machine-learning/data-production/classification/%s' % job_id
        elif environment == 'dev':
            user_name = os.environ['USER']
            if not user_name:
                logging.fatal(
                    'USER environment variable cannot be empty for dev runs.')
                sys.exit(-1)
            root_path = 'gs://machine-learning-dev-ttl-120d/data-production/classification/%s/%s' % (
                user_name, job_id)
        else:
            logging.fatal('Invalid environment: %s', env)
            sys.exit(-1)

        return GcpConfig(now, project_id, root_path)
