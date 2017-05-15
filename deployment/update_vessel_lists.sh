nohup sudo docker run --rm -v ~/.ssh:/root/.ssh -v ~/vessel-classification/logs:/app/logs update_vessel_lists python deployment/update_vessel_lists.py > /dev/null 2>&1 &
