FROM ubuntu:16.04

# Install java
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y  software-properties-common && \
    add-apt-repository ppa:webupd8team/java -y && \
    apt-get update && \
    echo oracle-java7-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
    apt-get install -y oracle-java8-installer && \
    apt-get clean

# Prepare dependencies
RUN apt-get update && \
  apt-get install -y apt-transport-https ca-certificates unzip curl libcurl3 wget

# Install SBT
RUN echo "deb https://dl.bintray.com/sbt/debian /" > /etc/apt/sources.list.d/sbt.list && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823 && \
    apt-get update && \
    apt-get install -y sbt


# Install protoc
RUN wget https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip -O /tmp/protoc.zip && \
    unzip /tmp/protoc.zip -d /usr/lib/protoc && \
    rm /tmp/protoc.zip && \
    ln -s /usr/lib/protoc/bin/protoc /usr/bin/protoc

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
RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl && \
    pip install google-api-python-client pyyaml pytz newlinejson python-dateutil yattag

ADD ./classification /app/classification
ADD ./pipeline /app/pipeline

WORKDIR /app

ENV USER="VESSEL_LIST_UPDATER"

# Run app.py when the container launches
# CMD ["python", "deployment/update_vessel_lists.py"]
