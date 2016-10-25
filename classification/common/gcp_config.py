import logging
import os
import sys


class GcpConfig(object):
    def __init__(self, project_id, root_path):
        self.project_id = project_id
        self.root_path = root_path

    def model_path(self):
        return self.root_path + '/models'


# TODO(alexwilson): This config is too hard-coded to our current setup. Move
# out to config files for greater flexibility. Note there is an equivalent to
# this in Commmon.scala which should remain in-sync.
def makeConfig(environment, job_id):
    project_id = "world-fishing-827"
    if environment == 'prod':
        root_path = 'gs://world-fishing-827/data-production/classification/%s' % job_id
    elif environment == 'dev':
        user_name = os.environ['USER']
        if not user_name:
            logging.fatal(
                'USER environment variable cannot be empty for dev runs.')
            sys.exit(-1)
        root_path = 'gs://world-fishing-827-dev-ttl30d/data-production/classification/%s/%s' % (
            user_name, job_id)
    else:
        logging.fatal('Invalid environment: %s', env)
        sys.exit(-1)

    return GcpConfig(project_id, root_path)