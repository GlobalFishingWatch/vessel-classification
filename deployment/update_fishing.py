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
import common
from common import this_dir, classification_dir, pipeline_dir, top_dir, treniformis_dir, logdir
from common import checked_call, log, job_status, status_at_completion, parse_id_from_sbt_output

logpath = os.path.join(logdir, "log-{}".format(str(datetime.datetime.utcnow()).replace(' ', '_')))


def download_weights_if_needed():
    if not os.path.exists('vessel_characterization.model.ckpt'):
        checked_call(['gsutil', 'cp', 
            'gs://world-fishing-827/data-production/classification/fishing_detection.model.ckpt', '.'],
            cwd=classification_dir)
    else:
        log("Using existing weights without updating")


def sharded_paths(range_start, range_end):
    paths = []
    day = range_start
    while day <= range_end:
        paths.append(
            '  - "gs://benthos-pipeline/data-production-740/classify-pipeline/classify-logistic/{:%Y-%m-%d}/*-of-*"'
                .format(day))
        day += datetime.timedelta(days=1)
    return paths


def generate_features(range_start, range_end):
    template = """
inputFilePatterns:
{paths}
knownFishingMMSIs: "../../treniformis/treniformis/_assets/GFW/FISHING_MMSI/KNOWN_AND_LIKELY/ANY_YEAR.txt"
anchoragesRootPath: "gs://world-fishing-827/data-production/classification/release-0.1.0/pipeline/output"
minRequiredPositions: 10
encounterMinHours: 3
encounterMaxKilometers: 0.5
"""

    paths = sharded_paths(range_start, range_end)
    config = template.format(paths='\n'.join(paths))

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
                                       --job-name=update_fishing_detection \
                                       --generate-model-features=true \
                                       --generate-encounters=true \
                                       --job-config={config_path}"'''.format(config_path=fp.name)

        log("Executing command:")
        log(command)
        log()

        output = checked_call([command], shell=True, cwd=pipeline_dir)

    return parse_id_from_sbt_output(output)


# determine_dates (allow to be specified)
# generate features for padded dates
# run inference for dates
# run annotation for dates

"""python -m classification.run_inference prod.fishing_detection \
            --root_feature_path gs://world-fishing-827/data-production/classification/release-0.1.3/pipeline/output/features \
            --inference_parallelism 128 \
            --feature_dimensions 15  \
            --model_checkpoint_path   ./updated-fishing-model.ckpt-21622  \
            --metadata_file training_classes.csv \
            --fishing_ranges_file combined_fishing_ranges.csv \
            --inference_results_path=./test-auto-fishing.json.gz \
            --start_date 2017-06-01 \
            --end_date 2017-06-01
""" # TODO: Need to update checkout / results / start and end dates. Maybe reduce parallelism too.


def run_generate_features(range_start, range_end):
    log("Starting Feature Generation")
    feature_id = generate_features(range_start, range_end)

    log("Waiting for Feature Generation Complete, ID:", feature_id)
    status = status_at_completion(feature_id)
    if status != 'DONE':
        raise RuntimeError("feature generation did not complete ({})".format(status))
    log("Feature Generation Complete")


def run_inference(start_date, end_date):
    command = """
        python -m classification.run_inference prod.fishing_detection \\
            --root_feature_path gs://world-fishing-827-dev-ttl30d/data-production/classification/{user}/update_vessel_lists/pipeline/output/features \\
            --inference_parallelism 64 \\
            --feature_dimensions 15 \\
            --inference_results_path ./update_fishing_detection.json.gz \\
            --model_checkpoint_path   ./fishing_detection.model.ckpt  \\
            --metadata_file training_classes.csv \\
            --fishing_ranges_file combined_fishing_ranges.csv
            --start_date {start_date:%Y-%m-%d}
            --end_date {end_date:%Y-%m-%d}
            """.format(user=os.environ.get('USER'), start_date=start_date, end_date=end_date)

    log("Running command:")
    log(command)

    checked_call([command], shell=True, cwd=classification_dir)


def run_annotation(start_date, end_date):
    template = """
inputFilePatterns:
  {paths}
knownFishingMMSIs: "../../treniformis/treniformis/_assets/GFW/FISHING_MMSI/KNOWN_LIKELY_AND_SUSPECTED/ANY_YEAR.txt"
jsonAnnotations:
  - inputFilePattern: "update_fishing_detection.json.gz"
    timeRangeFieldName: "fishing_localisation"
    outputFieldName: "nnet_score"
    defaultValue: 1.0
"""
    # annotate most recent two weeks.
    paths = sharded_paths(start_date, end_date)

    config = template.format(paths='\n'.join(paths))

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(config)
        fp.flush()
        log("Using Config:")
        log(config)
        log()

        command = ''' sbt aisAnnotator/"run --job-config={config_path} 
                                            --env=dev --job-name=annotate_all 
                                            --maxNumWorkers=100 
                                            --diskSizeGb=100 
                                            --output-path=gs://world-fishing-827/data-production/classification/incremental"
                                            '''.format(config_path=fp.name)

        log("Executing command:")
        log(command)
        log()

        output = checked_call([command], shell=True, cwd=pipeline_dir)

    return parse_id_from_sbt_output(output)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update Vessel Lists.')
    parser.add_argument('--skip-feature-generation', help='skip generating new features', action='store_true')
    parser.add_argument('--skip-inference', help='skip running inference', action='store_true')
    parser.add_argument('--skip-annotation', help='skip annotating pipeline data', action='store_true')
    args = parser.parse_args()

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=14)
    feature_start_date = start_date - datetime.timedelta(days=14)

    try:
        create_dirs_if_needed()

        download_weights_if_needed()

        if not args.skip_feature_generation:
            # Generate features for last 30 datas
            run_generate_features(feature_start_date, end_date)

        if not args.skip_inference:
            run_inference(start_date, end_date)

        if not args.skip_annotation:
            run_annotation(start_date, end_date)

    except Exception as err:
        log("Execution failed with:", repr(err))


