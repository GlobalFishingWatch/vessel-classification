# Global Fishing Watch Vessel Classification Pipeline.

[Global Fishing Watch](http://globalfishingwatch.org) is a partnership between [Skytruth](https://skytruth.org), [Google](https://environment.google/projects/fishing-watch/) and [Oceana](http://oceana.org) to map all of the trackable commercial fishing activity in the world, in near-real time, and make it accessible to researchers, regulators, decision-makers, and the public.

This repository contains code to build Tensorflow models to classify vessels and identify fishing behavior
based on [AIS](https://en.wikipedia.org/wiki/Automatic_identification_system) data.

(This is not an official Google Product).

## Overview

Use AIS, and possibly VMS data in the future, to extract various types of information including:
   
  - Vessel types

  - Vessel fishing activity

  - Vessel attributes (length, tonnage, etc)

The project consists of a convolutional neural networks (CNN) that infers vessel features.


### Neural Networks

We have two CNN in production, as well as several experimental nets. One net
predict vessel class (`longliner`, `cargo`, `sailing`, etc), as well as
vessel length and other vessel parameters, while the second predicts whether 
a vessel is fishing or not at a given time point.

*We initially used a single CNN to predict everything at once,
but we've moveed to having two CNN.  The original
hope was that we would be able to take advantage of transfer learning between
the various features. However, we did not see any gains from that, and using
a multiple nets adds useful flexibility.*

The nets share a similar structure, consisting of a large number (currently 9)
of 1-D convolutional layers, followed by a single dense layer. The net for 
fishing prediction is somewhat more complicated since it must predict fishing at
each point. To do this all of the layers of the net are combined, with upscaling
of the upper layers, to produce a set of features at each point. 
These design of these nets incorporates ideas are borrowed
from the ResNets and Inception nets, among other places, but adapted for the 1D environment.

The code associated with the neural networks is located in
`classification`. The models themselves are located
in `classification/models`. 

## Data layout

*The data layout is currently in flux as we move data generation to Python-Dataflow
managed by Airflow*

### Common parameters

In order to support the above layout, all our programs need the following common parameters:

* `env`: to specify the environment - either development or production.
* `job-name`: for the name (or date) of the current job.
* Additionally if the job is a dev job, the programs will read the $USER environment variable
  in order to be able to choose the appropriate subdirectory for the output data.


# Neural Net Classification

## Running Stuff

-  `python -m train.deploy_cloudml` -- launch a training run on cloudml. Use `--help` to see options

   If not running in the SkyTruth/GFW environment, you will need to edit `deploy_cloudml.yaml`
   to set the gcs paths correctly.

   For example, to run vessel classification in the dev environment with the name `test`:

      python -m train.deploy_cloudml \
              --env dev \
              --model_name vessel_characterization \
              --job_name test_deploy_v20200601 \
              --config train/deploy_v_py3.yaml \
              --feature_path gs://machine-learning-dev-ttl-120d/features/vessel_char_track_id_features_v20200428/features \
              --vessel_info char_info_tid_v20200428.csv \
              --fishing_ranges det_ranges_tid_v20200428.csv


   **IMPORTANT**: Even though there is a maximum number of training steps specified, the CloudML
   process does not shut down reliably.  You need to periodically check on the process and kill it
   manually if it has completed and is hanging. In addition, there are occasionally other problems
   where either the master or chief will hang or die so that new checkpoints aren't written, or
   new validation data isn't written out. Again, killing and restarting the training is the solution.
   (This will pick up at the last checkpoint saved.)

- *running training locally* -- this is primarily for testing as it will be quite slow unless you
  have a heavy duty machine:

        python -m classification.run_training \
            fishing_range_classification \
            --feature_dimensions 14 \
            --root_feature_path FEATURE_PATH \
            --training_output_path OUTPUT_PATH \
            --fishing_range_training_upweight 1 \
            --metadata_file VESSEL_INFO_FILE_NAME \
            --fishing_ranges_file FISHING_RANGES_FILE_NAME \
            --split {0, 1, 2, 3, 4, -1}
            --metrics minimal

- `python -m train.compute_metrics` -- evaluate results and dump vessel lists. Use `--help` to see options


* Inference is now run solely through Apache Beam. See README in pipe-features for details


## Local Environment Setup

* Python 3.7++
* Tensorflow version >1.14.0,<2.0 from (https://www.tensorflow.org/get_started/os_setup)
* `pip install google-api-python-client pyyaml pytz newlinejson python-dateutil yattag`






