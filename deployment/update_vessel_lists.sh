echo "Starting Update"

CLASSIFY_GCS_ROOT_PATH=$1
ANCHORGES_GCS_ROOT_PATH=$2
VESSEL_CHARACTERIZATION_MODEL_GCS_PATH=$3
FEATURES_GCS_ROOT_PATH=$4

nohup sudo docker run --rm -v ~/.ssh:/root/.ssh -v ~/vessel-classification/logs:/app/logs update_vessel_lists python deployment/update_vessel_lists.py --classify-gcs-root-path "$CLASSIFY_GCS_ROOT_PATH" --anchorages-gcs-root-path "$ANCHORGES_GCS_ROOT_PATH" --features-gcs-root-path "$FEATURES_GCS_ROOT_PATH" --vessel-characterization-model-gcs-path "$VESSEL_CHARACTERIZATION_MODEL_GCS_PATH" > ~/vessel-classification/logs/last_run.txt 2>&1 &