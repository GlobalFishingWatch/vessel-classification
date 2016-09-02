# Global Fishing Watch Vessel Classification Pipeline.

Vessel classification pipeline: feature generation and model training/inference.

[![Build Status](https://travis-ci.org/GlobalFishingWatch/vessel-classification-pipeline.svg?branch=master)](https://travis-ci.org/GlobalFishingWatch/vessel-classification-pipeline)

# Building

The various projects are built using the Scala build tool 'sbt'. You need a JVM on your machine
to get up and running. SBT has a repl, which can be entered using the checked-in 'sbt' script in
the root directory. Some commands:

* To compile: 'compile'.
* To run: 'run'.
* To test: 'test'.
* To autoformat the code before check-in: 'scalafmt'.
* To generate html Scaladoc: 'doc'.

SBT uses maven to handle it's dependencies. So the first time you attempt a build your machine
may take some time to download all the required libraries.

# Running jobs

* Compute Engine.
  * Install the SDK: https://cloud.google.com/sdk/docs/.
  * Sign in: `gcloud auth application-default login`.
* Cloud Dataflow
   * Run jobs, specifying the zone, e.g. `--zone=europe-west1-c`.
