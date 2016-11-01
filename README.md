# Global Fishing Watch Vessel Classification Pipeline.

Vessel classification pipeline: feature generation and model training/inference.

[![Build Status](https://travis-ci.org/GlobalFishingWatch/vessel-classification-pipeline.svg?branch=master)](https://travis-ci.org/GlobalFishingWatch/vessel-classification-pipeline)

# Data layout

The production classification pipeline has multiple stages, computing: feature generation, port
inference, encounters, model training, vessel type inference, fishing locality inference, accuracy
evaluation. This is implemented in several stages, each requiring outputs to GCS. In addition we
run experiments of our own to improve or develop the pipeline further. We need the data to be laid
out systematically on GCS. Currently we use the following structure:

* `world-fishing-827/data-production` (prod)
  * `classification`
    * `<date> or <job name>`
      * `pipeline` (for cloud dataflow pipeline output)
        * `staging` (for the various files required for runs, e.g. dataflow jars).
        * `output` (for the output from the dataflow pipeline).
      * `inference` (for the output from inference + the eval framework).
* `world-fishing-827-dev-ttl30d/data-production` (dev)
    * `classification`
      * `alex`
        * `<job name>`
          * A mirror of the files you see under `production`.
          * Plus a new directory `models`
      * `tim`
      * etc...

Here, we have a production pipeline running under (probably) cron, generating new results daily to
(`world-fishing-827/data-production`).
We have tight control over the code that's pushed to run on production (probably via a Docker image
registered on GCR).

We then have a dev directory (`world-fishing-827-dev-ttl30d/data-production`). Anything we're trying
that hasn't yet hit production will end up in a `dev/<username>` directory, mirroring the
directories in prod but isolated from prod and from other developer's experiments. This directory
has a TTL set on all contents such that anything that is older than 30 days will be automatically
deleted (to keep our GCS costs low and prevent infinite accumulation of experiments).

I further propose we have also have a subdirectory under `dev`: `models` (or some other name) where
we experiment with and train new models. Given our models are quite small, I would be inclined to
package them in the Docker images we deploy rather than store them on GCS. We could commit the
latest model files to git.

## Common parameters

In order to support the above layout, all our programs need the following common parameters:

* `env`: to specify the environment - either development or production.
* `job-name`: for the name (or date) of the current job.
* Additionally if the job is a dev job, the programs will read the $USER environment variable
  in order to be able to choose the appropriate subdirectory for the output data.

# Developing

* Code is to be formatted using [YAPF](https://github.com/google/yapf) before submission. See YAPF section below.


# Setup and building

## Summary of Requirements

* A JVM.
* A proto3-compatible version of protoc. See: [protocol buffers](https://developers.google.com/protocol-buffers/).
* Python.
* Tensorflow.
* [Docker](https://docs.docker.com).
  * For linux, follow the installation instructions on the Docker site, do not use the apt package.
* [Google Compute Engine](https://console.cloud.google.com) access and [SDK](https://cloud.google.com/sdk) installed locally.

## Scala

In subdirectory `scala`, the feature/ports/encounter pipeline.

The various projects are built using the Scala build tool `sbt`. SBT has a repl, which can be
entered using the checked-in `sbt` script in the root directory. Some commands:

* To compile: 'compile'.
* To run: 'run'.
* To test: 'test'.
* To autoformat the code before check-in: 'scalafmt'.
* To generate html Scaladoc: 'doc'.

SBT uses maven to handle it's dependencies. So the first time you attempt a build your machine
may take some time to download all the required libraries.

## Python

In subdirectory `python`, everything related to TF and our NN models plus evaluation.

Python programs have a few dependencies that can be installed using pip.

To install pip:

* `sudo apt-get install python-pip python-dev build-essential`
* `sudo easy_install pip`
* `sudo pip install --upgrade virtualenv`

To install TensorFlow, follow [these instructions](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#using-pip). For example for Linux, call:

* `sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl`

To install the dependencies:

* `sudo pip install google-api-python-client pyyaml`

## YAPF

[YAPF](https://github.com/google/yapf) is a code formatter for Python. All our python code should
be autoformatted with YAPF before committing. To install it, run:

* `sudo pip install yapf`

Run `yapf -r -i .` in the top level directory to fix the format of the full project.


## Deployment

Some of our jobs are run on managed services (for instance the feature pipeline on Cloud Dataflow, the
tensor flow model training on Cloud ML). But other jobs are deployed to Compute Engine using Docker.

To build and deploy inference, from the root directory:

* `docker build -f deploy/inference/Dockerfile .`


# Adding new models

* Create a directory in classification/classification/models with the model name (usually the developer name)
* Add the model to setup.py


# Running jobs

* Compute Engine.
  * Install the SDK: https://cloud.google.com/sdk/docs/.
  * Sign in: `gcloud auth application-default login`.
* Cloud Dataflow
   * From the sbt console:
   * Run jobs, specifying the zone and max number of workers, e.g.
     `run --zone=europe-west1-c  --maxNumWorkers=80 --job-name=new_pipeline_features --env=dev`.
* Running TF locally:
   * Training:
       - `python -m classification.run_training alex.vessel_classification <...>`
* Cloud ML
   * Training:
       - `./deploy_cloudml.py --model_name alex.vessel_classification --env dev --job_name test2`
       - `./deploy_cloudml.py --model_name tim.tmodel_1 --env dev --job_name test2`
