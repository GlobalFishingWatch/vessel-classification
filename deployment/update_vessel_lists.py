from __future__ import print_function
from __future__ import division

import sys
import subprocess
import json
import time
import os
import tempfile
import datetime
import shutil


this_dir = os.path.abspath(os.path.dirname(__file__))
classification_dir = this_dir
pipeline_dir = os.path.abspath(os.path.join(this_dir, '../pipeline'))
top_dir = os.path.abspath(os.path.join(this_dir, '../..'))
treniformis_dir = os.path.abspath(os.path.join(this_dir, '../../treniformis'))


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

def add_bot_key_if_present():
    if os.path.exists("~/.ssh/gfw-bot-key"):
        subprocess.check_output(["eval $(ssh-agent -s) && ssh-add ~/.ssh/gfw-bot-key"],
            shell=True, stderr=subprocess.STDOUT)

def clone_treniformis_if_needed():
    if not os.path.exists('../../treniformis'):
        subprocess.check_call(['git', 'clone', 'https://github.com/GlobalFishingWatch/treniformis.git'], cwd=top_dir)
    else:
        print("Using existing treniformis without updating")

def download_weights_if_needed():
    if not os.path.exists('vessel_characterization.model.ckpt'):
        subprocess.check_call(['gsutil', 'cp', 
            'gs://world-fishing-827/data-production/classification/vessel_characterization.model.ckpt', '.'],
            cwd=classification_dir)
    else:
        print("Using existing weights without updating")

def generate_features(range_end):
    # We need 180 days, but do more, just to be safe, we usually don't have 
    # data right up to today anyway.
    range_start = range_end - datetime.timedelta(days=220)

    template = """
inputFilePatterns:
{}
knownFishingMMSIs: "../../treniformis/treniformis/_assets/GFW/FISHING_MMSI/KNOWN_AND_LIKELY/ANY_YEAR.txt"
anchoragesRootPath: "gs://world-fishing-827/data-production/classification/release-0.1.0/pipeline/output"
minRequiredPositions: 100
encounterMinHours: 3
encounterMaxKilometers: 0.5
"""
    paths = []
    month = range_start.month
    year = range_start.year
    while month < range_end.month or year < range_end.year:
        paths.append(
            '  - "gs://benthos-pipeline/data-production-740/classify-pipeline/classify-logistic/{year:4d}-{month:02d}-*/*-of-*"'.format(
            year=year, month=month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    config = template.format('\n'.join(paths))


    with tempfile.NamedTemporaryFile() as fp:
        fp.write(config)
        fp.flush()
        print("Using Config:")
        print(config)
        print()

        command = '''sbt features/"run --env=dev \
                                       --zone=us-central1-f \
                                       --experiments=use_mem_shuffle \
                                       --workerHarnessContainerImage=dataflow.gcr.io/v1beta3/java-batch:1.9.0-mm  \
                                       --maxNumWorkers=100 \
                                       --job-name=update_vessel_lists \
                                       --generate-model-features=true \
                                       --generate-encounters=false \
                                       --job-config={config_path}"'''.format(config_path=fp.name)

        print("Executing command:")
        print(command)
        print()

        try:
            output = subprocess.check_output([command], shell=True, cwd=pipeline_dir, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            print("Call failed with this output:", err.output)
            raise

    return parse_id_from_sbt_output(output)

def run_generate_features(range_end):
    print("Starting Feature Generation")
    feature_id = generate_features(range_end)

    print("Waiting for Feature Generation Complete, ID:", feature_id)
    status = status_at_completion(feature_id)
    if status != 'DONE':
        raise RuntimeError("feature generation did not complete ({})".format(status))
    print("Feature Generation Complete")

def run_inference():
    command = """
        python -m classification.run_inference prod.vessel_characterization \\
            --root_feature_path gs://world-fishing-827-dev-ttl30d/data-production/classification/{}/update_vessel_lists/pipeline/output/features \\
            --inference_parallelism 128 \\
            --feature_dimensions 12 \\
            --inference_results_path ./update_vessel_lists.json.gz \\
            --model_checkpoint_path   ./vessel_characterization.model.ckpt  \\
            --metadata_file training_classes.csv \\
            --fishing_ranges_file combined_fishing_ranges.csv

    """.format(os.environ.get('USER'))
    print("Running command:")
    print(command)
    print()
    try:
        subprocess.check_output([command], shell=True, cwd=classification_dir, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print("Call failed with this output:", err.output)
        raise


def create_lists():
    command = """
        python compute_metrics.py     \\
            --inference-path update_vessel_lists.json.gz  \\
            --label-path classification/data/training_classes.csv \\
            --dest-path update_vessel_lists.html \\
            --fishing-ranges classification/data/combined_fishing_ranges.csv \\
            --skip-localisation-metrics \\
            --dump-labels-to updated-labels \
            --dump-attributes-to updated-attributes \
            --dump-years ALL_ONLY
        """
    print("Running command:")
    print(command)
    print()
    try:
        subprocess.check_output([command], shell=True, cwd=classification_dir, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print("Call failed with this output:", err.output)
        raise

def update_treniformis(date):
    # Assumes treniformis is installed alongside 
    if not os.path.exists(treniformis_dir):
        raise RuntimeError("Treniformis not installed in the correct place")
    dest_dir = os.path.join(treniformis_dir, 'treniformis', '_assets', 'GFW', 
                                'VESSEL_INFO', 'VESSEL_LISTS')
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    shutil.copy('updated-labels/ALL_YEARS.csv',
        os.path.join(dest_dir, 'LABELS_{}_{}_{}.csv'.format(date.year, date.month, date.day)))
    shutil.copy('updated-attributes/ALL_YEARS.csv',
        os.path.join(dest_dir, 'ATTRIBUTES_{}_{}_{}.csv'.format(date.year, date.month, date.day)))



# TODO: allow date to be passed in.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update Vessel Lists.')
    parser.add_argument('--skip-feature-generation', help='skip generating new features', action='store_true')
    parser.add_argument('--skip-inference', help='skip running inference', action='store_true')
    parser.add_argument('--skip-list-generation', help='skip running inference', action='store_true')
    parser.add_argument('--skip-update-treniformis', help='skip updating treniformis', action='store_true')
    args = parser.parse_args()

    date = datetime.date.today()

    add_bot_key_if_present()

    clone_treniformis_if_needed()

    download_weights_if_needed()

    if not args.skip_feature_generation:
        # Generate features for last six months
        run_generate_features(date)

    if not args.skip_inference:
        run_inference()

    if not args.skip_list_generation:
        create_lists()

    if not args.skip_update_treniformis:
        update_treniformis(date)


