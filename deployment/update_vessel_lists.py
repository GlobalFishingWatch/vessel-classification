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
from common import this_dir, classification_dir, pipeline_dir, top_dir, treniformis_dir, logdir
from common import checked_call, log, job_status, status_at_completion, parse_id_from_sbt_output
from common import gcs_base, clone_treniformis_if_needed
import common




def download_weights_if_needed():
    if not os.path.exists('vessel_characterization.model.ckpt'):
        checked_call(['gsutil', 'cp', 
            'gs://world-fishing-827/data-production/classification/vessel_characterization.model.ckpt', '.'],
            cwd=classification_dir)
    else:
        log("Using existing weights without updating")

def create_dirs_if_needed():
    label_path = os.path.join(classification_dir, 'updated-labels')
    if not os.path.exists(label_path):
        log("Adding path", label_path)
        os.mkdir(label_path)
    regression_path = os.path.join(classification_dir, 'updated-attributes')
    if not os.path.exists(regression_path):
        log("Adding path", regression_path)
        os.mkdir(regression_path)

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
            '  - "{base}{year:4d}-{month:02d}-*/*-of-*"'.format(
            base=gcs_base, year=year, month=month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    config = template.format('\n'.join(paths))


    with tempfile.NamedTemporaryFile() as fp:
        fp.write(config)
        fp.flush()
        log("Using Config:")
        log(config)
        log()

        command = '''sbt features/"run --env=dev \
                                       --zone=us-central1-f \
                                       --experiments=use_mem_shuffle \
                                       --workerHarnessContainerImage=dataflow.gcr.io/v1beta3/java-batch:1.9.0-mm  \
                                       --maxNumWorkers=100 \
                                       --job-name=update_vessel_lists \
                                       --generate-model-features=true \
                                       --generate-encounters=true \
                                       --job-config={config_path}"'''.format(config_path=fp.name)

        log("Executing command:")
        log(command)
        log()

        output = checked_call([command], shell=True, cwd=pipeline_dir)

    return parse_id_from_sbt_output(output)

def run_generate_features(range_end):
    log("Starting Feature Generation")
    feature_id = generate_features(range_end)

    log("Waiting for Feature Generation Complete, ID:", feature_id)
    status = status_at_completion(feature_id)
    if status != 'DONE':
        raise RuntimeError("feature generation did not complete ({})".format(status))
    log("Feature Generation Complete")

def run_inference():
    command = """
        python -m classification.run_inference prod.vessel_characterization \\
            --root_feature_path gs://world-fishing-827-dev-ttl30d/data-production/classification/{}/update_vessel_lists/pipeline/output/features \\
            --inference_parallelism 64 \\
            --feature_dimensions 15 \\
            --inference_results_path ./update_vessel_lists.json.gz \\
            --model_checkpoint_path   ./vessel_characterization.model.ckpt  \\
            --metadata_file training_classes.csv \\
            --fishing_ranges_file combined_fishing_ranges.csv

    """.format(os.environ.get('USER'))
    log("Running command:")
    log(command)

    checked_call([command], shell=True, cwd=classification_dir)


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
    log("Running command:")
    log(command)
    log()
    checked_call([command], shell=True, cwd=classification_dir)


def update_treniformis(date):
    # Assumes treniformis is installed alongside 
    if not os.path.exists(treniformis_dir):
        raise RuntimeError("Treniformis not installed in the correct place")
    dest_dir = os.path.join(treniformis_dir, 'treniformis', '_assets', 'GFW', 
                                'VESSEL_INFO', 'VESSEL_LISTS')
    log("Copying files to", dest_dir)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    shutil.copy('classification/updated-labels/ALL_YEARS.csv',
        os.path.join(dest_dir, 'LABELS_{}_{}_{}.csv'.format(date.year, date.month, date.day)))
    shutil.copy('classification/updated-attributes/ALL_YEARS.csv',
        os.path.join(dest_dir, 'ATTRIBUTES_{}_{}_{}.csv'.format(date.year, date.month, date.day)))
    command = "eval $(ssh-agent -s) && ssh-add ~/.ssh/gfw-bot-key && python scripts/update_dates_bump_version_and_commit.py"
    log("running command:", command)
    output = checked_call([command], cwd=treniformis_dir, shell=True)
    log(output)



# TODO: allow date to be passed in.

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update Vessel Lists.')
    parser.add_argument('--skip-feature-generation', help='skip generating new features', action='store_true')
    parser.add_argument('--skip-inference', help='skip running inference', action='store_true')
    parser.add_argument('--skip-list-generation', help='skip generating new lists', action='store_true')
    parser.add_argument('--skip-update-treniformis', help='skip updating treniformis with new lists', action='store_true')
    args = parser.parse_args()

    date = common.most_recent(gcs_base + "{:%Y-%m-%d}/*")

    try:
        create_dirs_if_needed()

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
    except Exception as err:
        log("Execution failed with:", repr(err))


