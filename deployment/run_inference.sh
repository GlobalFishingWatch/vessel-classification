#!/bin/bash
set -e

cd `dirname $0`
source common.sh

INFERENCE_IMAGE_TAG=vessel-classification-inference

echo "Building Docker image."
docker build -t "${INFERENCE_IMAGE_TAG}" -f inference/Dockerfile ..

echo "Tagging docker image."
docker tag "${INFERENCE_IMAGE_TAG}" gcr.io/"${GCP_PROJECT_ID}"/"${INFERENCE_IMAGE_TAG}"

echo "Pushing image to GCR."
gcloud docker push gcr.io/"${GCP_PROJECT_ID}"/"${INFERENCE_IMAGE_TAG}"

echo "Launching inference job on GCE."
gcloud alpha compute instances create-from-container "${INFERENCE_IMAGE_TAG}" \
  --docker-image=gcr.io/"${GCP_PROJECT_ID}"/"${INFERENCE_IMAGE_TAG}" \
  --zone=europe-west1-d --machine-type=n1-standard-16 \
  --run-command="echo Hello world"