from __future__ import print_function
from __future__ import division

import sys
import subprocess
import json
import time
import os
import tempfile
import datetime


this_dir = os.path.abspath(os.path.dirname(__file__))
classification_dir = this_dir
pipeline_dir = os.path.abspath(os.path.join(this_dir, '../pipeline'))


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

def generate_features():

    range_end = datetime.date.today()
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
                                       --job-name=update_vessel_lists --generate-model-features=true \
                                       --generate-encounters=false \
                                       --job-config={config_path}"'''.format(config_path=fp.name)

        print("Executing command:")
        print(command)
        print()

        output = subprocess.check_output([command], shell=True, cwd=pipeline_dir)

    return parse_id_from_sbt_output(output)

def run_generate_features():
    print("Starting Feature Generation")
    feature_id = generate_features()

    print("Waiting for Feature Generation Complete, ID:", feature_id)
    status = status_at_completion(feature_id)
    if status != 'DONE':
        raise RuntimeError("feature generation did not complete ({})".format(status))
    print("Feature Generation Complete")

def run_inference():
    command = """
        python -m classification.run_inference prod.vessel_characterization \\
            --root_feature_path gs://world-fishing-827/data-production/classification/release-0.1.2/pipeline/output/features \\
            --inference_parallelism 128 \\
            --feature_dimensions 12 \\
            --inference_results_path ./update_vessel_lists.json.gz \\
            --model_checkpoint_path   ./vessel_characterization.model.ckpt  \\
            --metadata_file training_classes.csv \\
            --fishing_ranges_file combined_fishing_ranges.csv

    """
    print("Running command:")
    print(command)
    print()
    subprocess.check_output([command], shell=True, cwd=classification_dir)


def create_lists():
    command = """
        python compute_metrics.py     \\
            --inference-path update_vessel_lists.json.gz  \\
            --label-path classification/data/training_classes.csv \\
            --dest-path update_vessel_lists.html \\
            --fishing-ranges classification/data/combined_fishing_ranges.csv \\
            --skip-localisation-metrics \\
            --dump-labels-to updated-labels \
            --dump-attributes-to updated-attributes
        """
    print("Running command:")
    print(command)
    print()
    subprocess.check_output([command], shell=True, cwd=classification_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update Vessel Lists.')
    parser.add_argument('--skip-feature-generation', help='skip generating new features', action='store_true')
    parser.add_argument('--skip-inference', help='skip running inference', action='store_true')
    parser.add_argument('--skip-list-generation', help='skip running inference', action='store_true')
    args = parser.parse_args()

    if not args.skip_feature_generation:
        # Generate features for last six months
        run_generate_features()

    if not args.skip_inference:
        run_inference()

    if not args.skip_list_generation:
        create_lists()



