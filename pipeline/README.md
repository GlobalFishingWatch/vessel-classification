# Pipeline

## Setup and building

### Summary of Requirements

* A JVM.
* [SBT](http://www.scala-sbt.org/), a Scala build tool.
* A proto3-compatible version of protoc. See: [protocol buffers](https://developers.google.com/protocol-buffers/).
* Python.
* Tensorflow.
* [Docker](https://docs.docker.com).
  * For linux, follow the installation instructions on the Docker site, do not use the apt package.
* [Google Compute Engine](https://console.cloud.google.com) access and [SDK](https://cloud.google.com/sdk) installed locally.

### Scala

In subdirectory `scala`, the feature/ports/encounter pipeline.

The various projects are built using the Scala build tool `sbt`. SBT has a repl, which can be
entered using by running the `sbt` command in the `pipeline` directory. Some commands:

* To compile: 'compile'.
* To run: 'run'.
* To test: 'test'.
* To autoformat the code before check-in: 'scalafmt'.
* To generate html Scaladoc: 'doc'.

SBT uses maven to handle it's dependencies. So the first time you attempt a build your machine
may take some time to download all the required libraries.


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


## Running jobs

* This is a recent command for running encounters.

   $ ./sbt
   > project features
   > run  --env=dev --zone=us-central1-f --experiments=use_mem_shuffle --workerHarnessContainerImage=dataflow.gcr.io/v1beta3/java-batch:1.8.0-mm --maxNumWorkers=200 --job-name=encounters --generate-model-features=false --generate-encounters=true --anchorages-root-path=gs://world-fishing-827/data-production/classification/release-0.1.0/pipeline/output --minRequiredPositions=0


 run  --env=dev --zone=us-central1-f --experiments=use_mem_shuffle --workerHarnessContainerImage=dataflow.gcr.io/v1beta3/java-batch:1.8.0-mm --maxNumWorkers=800 --job-name=encounters --generate-model-features=false --generate-encounters=true --anchorages-root-path=gs://world-fishing-827/data-production/classification/release-0.1.0/pipeline/output --minRequiredPositions=0

   run  --env=dev --zone=us-central1-f --workerHarnessContainerImage=dataflow.gcr.io/v1beta3/java-batch:1.8.0-mm --maxNumWorkers=200 --job-name=features --generate-model-features=true --generate-encounters=false --anchorages-root-path=gs://world-fishing-827/data-production/classification/release-0.1.0/pipeline/output --minRequiredPositions=100


This runs the annotator, for just 2016, but so far this is still crashing:

run --job-config=ais-annotator/config/2016_annotation.yaml --env=prod --job-name=paper_prep --maxNumWorkers=200  --yearsToRun=2016 --zone=us-central1-f



OLD:
* Cloud Dataflow
   * From the sbt console:
   * Run jobs, specifying the zone and max number of workers, e.g.
       - Anchorages: `run --env=dev --zone=europe-west1-c --job-name=anchoragesOnly --maxNumWorkers=600 --diskSizeGb=100`.
       - Feature pipeline: `run --env=dev --zone=europe-west1-c  --maxNumWorkers=80 --job-name=new_pipeline_features`.

