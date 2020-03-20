FROM python:3.7-stretch

RUN mkdir -p /opt/project
WORKDIR /opt/project

# Prepare dependencies
RUN apt-get update && \
  apt-get install -y apt-transport-https ca-certificates unzip curl libcurl3 wget

# Install APT dependencies
RUN apt-get -y update && \
    apt-get -y install python python-setuptools python-dev build-essential git

# Install google cloud
RUN curl -sSL https://sdk.cloud.google.com | bash && \
  /root/google-cloud-sdk/bin/gcloud config set --installation component_manager/disable_update_check true && \
  /root/google-cloud-sdk/bin/gcloud components install beta
ENV PATH $PATH:/root/google-cloud-sdk/bin

# Install python dependencies
RUN easy_install pip && \
    pip install --upgrade pip
RUN pip install tensorflow && \
    pip install google-api-python-client pyyaml pytz newlinejson python-dateutil yattag pandas-gbq && \
    pip install git+https://github.com/GlobalFishingWatch/bqtools.git

COPY . /opt/project
RUN pip install .
