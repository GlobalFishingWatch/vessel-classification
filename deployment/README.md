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

`gcloud compute --project "world-fishing-827" ssh --zone "europe-west1-d" "nnet-auto-test" --command 'bash ~/vessel-classification/deployment/update_vessel_lists.sh'`

Logs are placed in `vessel-classification\logs`
