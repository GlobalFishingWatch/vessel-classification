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
from common import gcs_base, clone_treniformis_if_needed

logpath = os.path.join(logdir, "log-{}".format(str(datetime.datetime.utcnow()).replace(' ', '_')))


def download_weights_if_needed():
    if not os.path.exists('vessel_characterization.model.ckpt'):
        checked_call(['gsutil', 'cp', 
            'gs://world-fishing-827/data-production/classification/fishing_detection.model.ckpt', '.'],
            cwd=classification_dir)
    else:
        log("Using existing weights without updating")


def successfully_completed_one_of(job_ids, sleep_time=10): # TODO: add timeout
    while True:
        for job_id in job_ids:
            status = job_status(job_id)['currentState'].rsplit('_')[-1]
            if status == 'DONE':
                return job_id
            elif status in ('FAILED', 'CANCELLED'):
                raise RuntimeError("Annotation for {} did not complete ({})".format(job_id, status))
        time.sleep(sleep_time)

def upload_inference_results():
    destination = "gs://world-fishing-827-dev-ttl30d/data-production/classification/FISHING_UPDATER/update_fishing_detection.json.gz"
    log("Copying weights to", destination)
    checked_call(['gsutil', 'cp', 'update_fishing_detection.json.gz', destination],
        cwd=classification_dir)

def sharded_paths(range_start, range_end):
    paths = []
    day = range_start
    while day <= range_end:
        pth = '{base}{day:%Y-%m-%d}/*-of-*'.format(base=gcs_base, day=day)
        if common.exists_on_gcs(pth):
            paths.append(
                ('  - "{}"'.format(pth)), day)
        else:
            log("Skipping path missing from GCS:", pth)
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
    log("Generating paths for features")
    paths = [p for (p, d) in sharded_paths(range_start, range_end)]

    log("Generating config text for features")
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
            --root_feature_path gs://world-fishing-827-dev-ttl30d/data-production/classification/{user}/update_fishing_detection/pipeline/output/features \\
            --inference_parallelism 64 \\
            --feature_dimensions 15 \\
            --inference_results_path ./update_fishing_detection.json.gz \\
            --model_checkpoint_path   ./fishing_detection.model.ckpt  \\
            --metadata_file training_classes.csv \\
            --fishing_ranges_file combined_fishing_ranges.csv \\
            --start_date {start_date:%Y-%m-%d} \\
            --end_date {end_date:%Y-%m-%d}
            """.format(user=os.environ.get('USER'), start_date=start_date, end_date=end_date)

    log("Running command:")
    log(command)

    checked_call([command], shell=True, cwd=classification_dir)



# TODO: Annotate day-by-day
# TODO: Check for and remove old days

output_template = "gs://world-fishing-827/data-production/classification/incremental/{day:%Y-%m-%d}"
clobber_template = os.path.join(output_template, "*-of-*")

def run_annotation(start_date, end_date):
    template = """
inputFilePatterns:
{paths}
knownFishingMMSIs: "../../treniformis/treniformis/_assets/GFW/FISHING_MMSI/KNOWN_LIKELY_AND_SUSPECTED/ANY_YEAR.txt"
jsonAnnotations:
  - inputFilePattern: "gs://world-fishing-827-dev-ttl30d/data-production/classification/FISHING_UPDATER/update_fishing_detection.json.gz"
    timeRangeFieldName: "fishing_localisation"
    outputFieldName: "nnet_score"
    defaultValue: 1.0
"""
    # annotate most recent two weeks.
    paths = sharded_paths(start_date, end_date)

    active_ids = set()

    job_time = datetime.datetime.utcnow()

    for i, (p, d) in enumerate(paths):

        # start up to 10 workers and wait till one finishes to start another

        datestr = os.path.split(os.path.split(p)[0])[1]

        log("Anotating", datestr)

        config = template.format(paths=p)

        output_path = output_template.format(day=d)
        clobber_path = clobber_template.format(day=d)

        log("Removing existing files from", clobber_path)
        subprocess.call(['gsutil', '-m', 'rm', clobber_path])

        with tempfile.NamedTemporaryFile() as fp:
            fp.write(config)
            fp.flush()
            log("Using Config:")
            log(config, '\n')

            command = ''' sbt aisAnnotator/"run --job-config={config_path} \
                                                --env=dev \
                                                --job-name=annotate{job_time:%Y%m%d%H%M%S}{i} \
                                                --maxNumWorkers=5 \
                                                --diskSizeGb=100 \
                                                --output-path={output_path}" \
                                                '''.format(config_path=fp.name, output_path=output_path, job_time=job_time, i=i)

            log("Executing command:")
            log(command, '\n')

            output = checked_call([command], shell=True, cwd=pipeline_dir)

        annotation_id = parse_id_from_sbt_output(output)

        log("Started annotation with ID:", annotation_id)

        active_ids.add(annotation_id)

        while len(active_ids) >= 20:
            pid = successfully_completed_one_of(active_ids) 
            active_ids.remove(pid)
            log("Annotation Completed", pid)

    while len(active_ids):
        pid = successfully_completed_one_of(active_ids)
        active_ids.remove(pid)
        log("Annotation Complete for", pid)

    log("Annotation Fully Complete")


ALL = object()

def int_or_all(x):
    if x == "ALL":
        return ALL
    return int(x)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Update Vessel Lists.')
    parser.add_argument('--duration', type=int)
    parser.add_argument('--skip-feature-generation', help='skip generating new features', action='store_true')
    parser.add_argument('--skip-inference', help='skip running inference', action='store_true')
    parser.add_argument('--skip-annotation', help='skip annotating pipeline data', action='store_true')
    args = parser.parse_args()

    end_date = common.most_recent(gcs_base + "{day:%Y-%m-%d}/*")
    log("Using", end_date, "for end date")
    if args.duration is None:
        start_date = common.most_recent(clobber_template) - datetime.timedelta(days=7)
    else: 
        start_date = end_date - datetime.timedelta(days=args.duration)

    log("Using", start_date, "for start date")

    1/0

    feature_start_date = start_date - datetime.timedelta(days=14)

    try:
        download_weights_if_needed()

        clone_treniformis_if_needed()

        if not args.skip_feature_generation:
            # Generate features for last 30 datas
            run_generate_features(feature_start_date, end_date)

        if not args.skip_inference:
            run_inference(start_date, end_date)
            upload_inference_results()

        if not args.skip_annotation:
            run_annotation(start_date, end_date)

    except Exception as err:
        log("Execution failed with:", repr(err))


