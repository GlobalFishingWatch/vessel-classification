# Setup For GFW

* Create a Compute Engine instance (16 core is a good size). Give it full Cloud permissions.

* On the instance:
    - Install docker (https://docs.docker.com/engine/installation/linux/debian/#install-docker-ce)

    - Copy gfw-bot-key to ~/.ssh in the instance 
   
    - Set permission for key `chmod 400 ~/.ssh/gfw-bot-key`

    - Add github to known_hosts `ssh-keyscan -H github.com >> ~/.ssh/known_hosts`

    - Clone https://github.com/GlobalFishingWatch/vessel-classification

    - `cd vessel-classification`

    - Run `sudo docker build -f deployment/Dockerfile -t update_vessel_lists .` to create a docker instance.

# Running updates remotely.

* Vessel Lists

    gcloud compute --project "world-fishing-827" ssh --zone "us-east1-d" "nnet-inference-2" --command 'bash ~/vessel-classification/deployment/update_vessel_lists.sh'

* Fishing Detection

    gcloud compute --project "world-fishing-827" ssh --zone "us-east1-d" "nnet-inference-2" --command 'bash ~/vessel-classification/deployment/update_fishing.sh'

Logs are placed in `vessel-classification\logs`

# Debugging. 

## To run full update from the instance

`sudo docker run --rm -v ~/.ssh:/root/.ssh -v ~/vessel-classification/logs:/app/logs update_vessel_lists python deployment/update_vessel_lists.py`

## To run manually

`sudo docker run -it -v ~/.ssh:/root/.ssh -v ~/vessel-classification/logs:/app/logs update_vessel_lists bash`

Then for options:

`python deployment/update_vessel_lists.py --help` 

Now can run just the failing part for sped up debugging. For instance to just run inference:

`python deployment/update_vessel_lists.py --skip-feature-generation --skip-list-generation --skip-update-treniformis`




