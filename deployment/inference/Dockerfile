FROM ubuntu:16.04
RUN apt-get -y update
RUN apt-get -y install python python-pip python-dev build-essential
RUN easy_install pip
RUN pip install --upgrade
RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
RUN pip install google-api-python-client pyyaml pytz
ADD classification/classification classification
