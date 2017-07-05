from __future__ import print_function
import os
import subprocess
import datetime
import json
import time

this_dir = os.path.abspath(os.path.dirname(__file__))
classification_dir = os.path.abspath(os.path.join(this_dir, '../classification'))
pipeline_dir = os.path.abspath(os.path.join(this_dir, '../pipeline'))
top_dir = os.path.abspath(os.path.join(this_dir, '../..'))
treniformis_dir = os.path.abspath(os.path.join(top_dir, 'treniformis'))
logdir = os.path.abspath(os.path.join(this_dir, '../logs'))

logpath = os.path.join(logdir, "log-{}".format(str(datetime.datetime.utcnow()).replace(' ', '_')))

gcs_base = "gs://p_p429_resampling_3/data-production/classify-pipeline/classify/"

def exists_on_gcs(path):
    return not subprocess.call(["gsutil", "-q", "stat", path])


def most_recent(path_template, limit=100):
    today = datetime.date.today()
    for offset in range(limit):
        day = today - datetime.timedelta(days=offset)
        if exists_on_gcs(path_template.format(day)):
            return day

            
def checked_call(commands, **kwargs):
    kwargs['stderr'] = subprocess.STDOUT
    try:
        return subprocess.check_output(commands, **kwargs)
    except subprocess.CalledProcessError as err:
        log("Call failed with this output:", err.output)
        raise


def log(*args, **kwargs):
    """Just like 'print(), except that also outputs
       to the file located at `logpath'
    """
    args = (str(datetime.datetime.now()), ':') + args
    print(*args, **kwargs)
    with open(logpath, 'a') as f:
        kwargs['file'] = f
        print(*args, **kwargs)


def job_status(job_id):
    return json.loads(subprocess.check_output(['gcloud', '--format=json', 'dataflow',
                     'jobs', 'describe', job_id]))

def status_at_completion(job_id, sleep_time=10): # TODO: add timeout
    while True:
        status = job_status(job_id)['currentState'].rsplit('_')[-1]
        if status in ('DONE', 'FAILED', 'CANCELLED'):
            return status
        time.sleep(sleep_time)

def parse_id_from_sbt_output(output):
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('Submitted job:'):
            _, job_id = line.rsplit(None, 1)
            return job_id  