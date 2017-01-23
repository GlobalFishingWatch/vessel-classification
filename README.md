# Global Fishing Watch Vessel Classification Pipeline.

**Vessel classification pipeline: feature generation and model training/inference.**

[![Build Status](https://travis-ci.org/GlobalFishingWatch/vessel-classification-pipeline.svg?branch=master)](https://travis-ci.org/GlobalFishingWatch/vessel-classification-pipeline)

## Overview

Use AIS, and possibly VMS data in the future, to extract various types of information including:
   
  - Vessel types

  - Vessel fishing activity

  - Vessel attributes (length, tonnage, etc)

  - Vessel encounters

  - Port locations


The project consists of two parts, a Scala pipleline that ingests and processes AIS data and convolutional
neural networks (CNN) that infers vessel features.

### Scala Pipeline

The original purpose of the Scala pipeline was to extract features for the CNN. However it is now also
used to produces lists of vessel encounters and port locations. 

The pipeline generates features by first thinning the points to a density of at most one per 5 minutes, then 
collecting the following features at each point:

  - timestamp
  - time-delta (time difference between current and previous point)
  - distance-delta (distance between current and previous point)
  - speed (reported speed at current point)
  - inferred-speed (speed inferred from change in lat/lon)
  - course-delta (change in reported course relative to previous point)
  - local-time-of-day
  - local-month-of-year
  - inferred-course-delta (change in course inferred from lat/lon points between 
                            previous and current, and current and next points)
  - distance-to-shore
  - distance-to-bounding-anchorage (distance to nearest location that appears to be an anchorage)
  - time-to-bounding-anchorage (time till arrival / departure at nearest location that appears to
                                 be an anchorage)
  - num-neighbors (number of nearby vessels)

[comment]: # (TODO: add more info on encounters) 

[comment]: # (TODO: add more info on port locations) 

The code associated with the Scala pipeline is located under `vessel-classification-pipeline/pipeline`
and a good place to start examining the code is `vessel-classification-pipeline/pipeline/src/main/scala`.


### Neural Networks

We have three CNN in production, as well as several experimental nets. One net
predict vessel class (`longliner`, `cargo`, `sailing`, etc), the second predicts
vessel length, while the third predicts whether a vessel is fishing or not at
a given time point.

*We initially used a single CNN to predict everything at once,
we're moving to having one CNN for each predicted value.  The original
hope was that we would be able to take advantage of transfer learning between
the various features. However, we did not see any gains from that, and using
a different net for each feature add useful flexibility.*

The nets share a similar structure, consisting of a large number (currently 9)
of 1-D convolutional layers, followed by a single dense layer. The net for 
fishing prediction is somewhat more complicated since it must predict fishing at
each point. To do this all of the layers of the net are combined, with upscaling
of the upper layers, to produce. These design of these nets incorporate ideas are borrowed
from the ResNets and Inception nets among other places but adapted for the 1D environment.

The code associated with the neural networks are associated is located under
`vessel-classification/classification`. The models 



# Technical Details

## Data layout

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

### Common parameters

In order to support the above layout, all our programs need the following common parameters:

* `env`: to specify the environment - either development or production.
* `job-name`: for the name (or date) of the current job.
* Additionally if the job is a dev job, the programs will read the $USER environment variable
  in order to be able to choose the appropriate subdirectory for the output data.

## Developing

* Code is to be formatted using [YAPF](https://github.com/google/yapf) before submission. See YAPF section below.


## Setup and building

### Summary of Requirements

* A JVM.
* A proto3-compatible version of protoc. See: [protocol buffers](https://developers.google.com/protocol-buffers/).
* Python.
* Tensorflow.
* [Docker](https://docs.docker.com).
  * For linux, follow the installation instructions on the Docker site, do not use the apt package.
* [Google Compute Engine](https://console.cloud.google.com) access and [SDK](https://cloud.google.com/sdk) installed locally.

### Scala

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

### Python

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

### YAPF

[YAPF](https://github.com/google/yapf) is a code formatter for Python. All our python code should
be autoformatted with YAPF before committing. To install it, run:

* `sudo pip install yapf`

Run `yapf -r -i .` in the top level directory to fix the format of the full project.


### Deployment

Some of our jobs are run on managed services (for instance the feature pipeline on Cloud Dataflow, the
tensor flow model training on Cloud ML). But other jobs are deployed to Compute Engine using Docker.

To build a fat jar for any of the pipelines, we use an sbt plugin: 'sbt-assembly'.

* To build a fat jar of the feature pipeline (in sbt console):
  - `project features`.
  - `assembly`.
  - Once done, assembly will report the output path of the fat jar.

To build and deploy inference, from the root directory:

* `docker build -f deploy/inference/Dockerfile .`


## Adding new models

* Create a directory in classification/classification/models with the model name (usually the developer name)
* Add the model to setup.py

## Running jobs


* Cloud Dataflow
   * From the sbt console:
   * Run jobs, specifying the zone and max number of workers, e.g.
       - Anchorages: `run --env=dev --zone=europe-west1-c --job-name=anchoragesOnly --maxNumWorkers=600 --diskSizeGb=100`.
       - Feature pipeline: `run --env=dev --zone=europe-west1-c  --maxNumWorkers=80 --job-name=new_pipeline_features`.
* Running TF locally:
   * Training:
       - `python -m classification.run_training alex.vessel_classification <...>`
* Cloud ML Training:
       - `./deploy_cloudml.py --model_name alex.vessel_classification --env dev --job_name test2`
       - `./deploy_cloudml.py --model_name tim.tmodel_1 --env dev --job_name test2`
* Running Inference on Compute Engine (See below for setting up inference instance)
       - Start tmux: `tmux` or `tmux attach` depending if you already have a tmux session.
       - `cd ~/classifiction`
       # ...

       - Vessel classification: `python -m classification.run_inference alex.vessel_classification --root_feature_path gs://alex-dataflow-scratch/features-scratch/new_features_ports_anchorage/20161018T153546Z/features --inference_results_path=`pwd`/vessel_classification.json.gz --inference_parallelism 20  --feature_dimensions 11 --model_checkpoint_path vessel_classification_model.ckpt-500001`
       - Fishing localisation: `python -m classification.run_inference alex.fishing_range_classification --root_feature_path gs://alex-dataflow-scratch/features-scratch/new_features_ports_anchorage/20161018T153546Z/features --inference_results_path=`pwd`/fishing_localisation.json.gz --inference_parallelism 20  --feature_dimensions 11 --model_checkpoint_path vessel_classification_model.ckpt-500001`


* Setting up Compute Engine for Inference.
  * Install the SDK: https://cloud.google.com/sdk/docs/.
  * Sign in: `gcloud auth application-default login`.
  * Create an instance:
      - Need at least 8 cores; here is the command to create a 16 core machine:

            gcloud compute instances create nnet-inference --zone=us-east1-d --machine-type=n1-standard-16

      - SSH into the machine:

            gcloud compute ssh nnet-inference --zone=us-east1-d

      - Install and activate `tmux`:

            sudo apt-get -y update
            sudo apt-get install -y tmux
            tmux

      - Install other dependencies:

            sudo apt-get -y install python python-pip python-dev build-essential git virtualenv
            sudo easy_install pip
            sudo pip install --upgrade pip
            sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
            sudo pip install google-api-python-client pyyaml pytz newlinejson python-dateutil
            git clone https://github.com/GlobalFishingWatch/vessel-classification-pipeline.git


