# Global Fishing Watch Vessel Classification Pipeline.

[Global Fishing Watch](http://globalfishingwatch.org) is a partnership between [Skytruth](https://skytruth.org), [Google](https://environment.google/projects/fishing-watch/) and [Oceana](http://oceana.org) to map all of the trackable commercial fishing activity in the world, in near-real time, and make it accessible to researchers, regulators, decision-makers, and the public.

This repository contains code to process [AIS](https://en.wikipedia.org/wiki/Automatic_identification_system) data to produce features and to build Tensorflow models to classify vessels and identify fishing behaviour.

(This is not an official Google Product).

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
More details on the pipeline and instructions for running it are located in
`vessel-classification-pipeline/pipeline/README.md`


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

The code associated with the neural networks is located in
`vessel-classification/classification`. The models themselves are located
in `vessel-classification/classification/classification/models`. More details
on the neural network code and instructions for running it is located in 
`vessel-classification-pipeline/classification/README.md`


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



