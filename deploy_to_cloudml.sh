#!/bin/bash
PROJECT_ID=`gcloud config list project --format="value(core.project)"`
JOB_NAME=alex_vessel_classification
TRAIN_PATH=gs://${PROJECT_ID}-ml/${JOB_NAME}

python setup.py sdist --format=gztar

gsutil cp dist/alex_vessel_classification-1.0.tar.gz ${TRAIN_PATH}/alex_vessel_classification-1.0.tar.gz
echo "Deployed alex_vessel_classification to ${TRAIN_PATH}"

gcloud beta ml jobs submit training alex_vessel_classification_`date '+%Y%m%d_%H%M%S'` --config cloudml_deploy.yaml