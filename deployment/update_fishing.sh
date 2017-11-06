echo "Starting Update"
nohup sudo docker run --rm \
            -v ~/.ssh:/root/.ssh -v ~/vessel-classification/logs:/app/logs \
            -e SOURCE_COMMIT=`git rev-parse --short HEAD` \
            -v ~/vessel-classification/logs:/app/source update_vessel_lists \
            python deployment/update_fishing.py \
            > ~/vessel-classification/logs/last_fishing_run.txt 2>&1 &


