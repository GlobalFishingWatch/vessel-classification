# Neural Net Classification

## Running Stuff

-  `deploy_cloudml.py` -- launch a training run on cloudml. Use `--help` to see options

   If not running in the SkyTruth/GFW environment, you will need to edit `deploy_cloudml.yaml`
   to set the gcs paths correctly.

   For example, to run vessel classification in the dev environment with the name `test`:

        `./deploy_cloudml.py --model_name prod.vessel_classification --env dev --job_name test`

   **IMPORTANT**: Even though there is a maximum number of training steps specified, the CloudML
   process does not shut down reliably.  You need to periodically check on the process and kill it
   manually if it has completed and is hanging. In addition, there are occasionally other problems
   where either the master or chief will hang or die so that new checkpoints aren't written, or
   new validation data isn't written out. Again, killing and restarting the training is the solution.
   (This will pick up at the last checkpoint saved.)

- *running training locally* -- this is primarily for testing as it will be quite slow unless you
  have a heavy duty machine:

        python -m classification.run_training \
            prod.fishing_range_classification \
            --feature_dimensions 12 \
            --root_feature_path FEATURE_PATH \
            --training_output_path OUTPUT_PATH \
            --fishing_range_training_upweight 1 \
            --metadata_file training_classes.csv \
            --fishing_ranges_file combined_fishing_ranges.csv \
            --metrics minimal


- `compute_metrics.py` -- evaluate restults and dump vessel lists. Use `--help` to see options


- *running inference* -- Unless you have local access to a heavy duty machine, you should
  probably run this on ComputeEngine as described below. If running remotely, use tmux so 
  that your run doesn't die if your connection gets dropped.

   - Copy a model checkpoint locally:

      gsutil cp GCS_PATH_TO_CHECKPOINT  ./model.ckpt

   - Run inference job:

    * *Vessel Classification*. This command only infers result for only the test data 
      (for evaluation purposes), and infers a seperarate classification every 6 months:

       python -m classification.run_inference prod.vessel_classification \
              --root_feature_path GCS_PATH_TO_FEATURES \
              --inference_parallelism 32 \
              --feature_dimensions 12 \
              --dataset_split Test \
              --inference_results_path=./RESULT_NAME.json.gz \
              --model_checkpoint_path ./model.ckpt \
              --metadata_file training_classes.csv \
              --fishing_ranges_file combined_fishing_ranges.csv \
              --interval_months 6

   - *Fishing localisation*: This infers all fishing at all time points (no `--dataset_split` specification)

         python -m classification.run_inference prod.fishing_range_classification \
                --root_feature_path GCS_PATH_TO_FEATURES \
                --inference_parallelism 32 \
                --feature_dimensions 12 \
                --inference_results_path=./RESULT_NAME.json.gz \
                --model_checkpoint_path ./model.ckpt \
                --metadata_file training_classes.csv \
                --fishing_ranges_file combined_fishing_ranges.csv



## Local Environment Setup

* Python 2.7+
* Tensorflow 12.1 from (https://www.tensorflow.org/get_started/os_setup)
* `pip install google-api-python-client pyyaml pytz newlinejson python-dateutil yattag`

## ComputeEngine Setup for Inference 

* Install the SDK: https://cloud.google.com/sdk/docs/.
* Sign in: `gcloud auth application-default login`.
* Create an instance:
      - Need at least 8 cores; here is the command to create a 16 core machine named 'nnet-inference'
        in the 'us-east1-d' zone:

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
        sudo pip install google-api-python-client pyyaml pytz newlinejson python-dateutil yattag
        git clone https://github.com/GlobalFishingWatch/vessel-classification-pipeline.git

## Adding new models

* For development: create a directory in `classification/classification/models/dev` with the model name 
  (usually the developer name).  A `__init__.py` is required for the model to be picked up and the model
  package directory must be added to `setup.py`.

* For production: add the model to `classification/classification/models/prod`


## Formatting

[YAPF](https://github.com/google/yapf) is a code formatter for Python. All our python code should
be autoformatted with YAPF before committing. To install it, run:

* `sudo pip install yapf`

Run `yapf -r -i .` in the top level directory to fix the format of the full project.




