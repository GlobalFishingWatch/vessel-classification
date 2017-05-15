# Setup For GFW

* Create a Compute Engine instance (16 core is a good size). Give it full Cloud permissions.

* On the instance:
    - Install docker

    - Copy gfw-bot-key to ~/.ssh in the instance

    - Clone vessel classification

    - Run `sudo docker build -f deployment/Dockerfile -t update_vessel_lists .` to create a docker instance.

# Running updates remotely.

`gcloud compute --project "world-fishing-827" ssh --zone "europe-west1-d" "nnet-auto-test" --command 'bash ~/vessel-classification/deployment/update_vessel_lists.sh'`

Logs are placed in `vessel-classification\logs`