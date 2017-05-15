# Setup For GFW

* Create a Compute Engine instance (16 core is a good size). Give it full Cloud permissions.

* On the instance:
    - Install docker

    - Copy gfw-bot-key to ~/.ssh in the instance

    - Clone vessel classification

    - Run `sudo docker build -f deployment/Dockerfile -t update_vessel_lists .` to create a docker instance.

# Running updates remotely.

`gcloud compute --project "world-fishing-827" ssh --zone "europe-west1-d" "nnet-auto-test" --command 'sudo docker run --rm -v ~/.ssh:/root/.ssh -v ~/vessel-classification/logs:/app/logs update_vessel_lists python deployment/update_vessel_lists.py'`

Logs are placed in `vessel-classification\logs`