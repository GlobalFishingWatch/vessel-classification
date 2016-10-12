# Global Fishing Watch Vessel Classification Pipeline.

Vessel classification pipeline: feature generation and model training/inference.

[![Build Status](https://travis-ci.org/GlobalFishingWatch/vessel-classification-pipeline.svg?branch=master)](https://travis-ci.org/GlobalFishingWatch/vessel-classification-pipeline)

# Developing

* Code is to be formatted using [YAPF](https://github.com/google/yapf) before submission. Use `yapf -r -i .` in top level
  directory.


# Building

## Requirements

* A JVM.
* A proto3-compatible version of protoc. See: [protocol buffers](https://developers.google.com/protocol-buffers/).
* Python.
* Tensorflow.
* [Docker](https://docs.docker.com).
  * For linux, follow the instructions, do not use the apt package.
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

## Deployment

Some of our jobs are run on managed services (for instance the feature pipeline on Cloud Dataflow, the
tensor flow model training on Cloud ML). But other jobs are deployed to Compute Engine using Docker.

To build and deploy inference, from the root directory:

* `docker build -f deploy/inference/Dockerfile .`


# Running jobs

* Compute Engine.
  * Install the SDK: https://cloud.google.com/sdk/docs/.
  * Sign in: `gcloud auth application-default login`.
* Cloud Dataflow
   * From the sbt console:
   * Run jobs, specifying the zone and max number of workers, e.g. `run --zone=europe-west1-c  --maxNumWorkers=80`.
* Running TF locally:
   * Training:
       - `python -m classification.run_training alex.vessel_classification <...>`
* Cloud ML
   * Training:
       - `./deploy_cloudml.py alex.vessel_classification`
       - `./deploy_cloudml.py tim.tmodel_1`
