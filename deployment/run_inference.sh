#!/bin/bash
# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
