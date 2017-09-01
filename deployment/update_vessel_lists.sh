echo "Starting Update"
nohup sudo docker run --rm \
                -v ~/.ssh:/root/.ssh  
                -e SOURCE_COMMIT=`git rev-parse --short HEAD` \
                -v ~/vessel-classification/logs:/app/logs update_vessel_lists \
                python deployment/update_vessel_lists.py \
                > ~/vessel-classification/logs/last_run.txt 2>&1 &
