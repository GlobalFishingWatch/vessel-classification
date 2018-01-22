# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools
import glob
import os


DEPENDENCIES = [
    "ujson",
    "NewlineJSON"
]

# Frozen dependencies for the google cloud dataflow dependency
DATAFLOW_PINNED_DEPENDENCIES = [
    "apache-beam==2.1.0",
    "avro==1.8.2",
    "cachetools==2.0.1",
    "certifi==2017.7.27.1",
    "chardet==3.0.4",
    "crcmod==1.7",
    "dill==0.2.6",
    "enum34==1.1.6",
    "funcsigs==1.0.2",
    "future==0.16.0",
    "futures==3.1.1",
    "gapic-google-cloud-pubsub-v1==0.15.4",
    "google-apitools==0.5.11",
    "google-auth==1.1.0",
    "google-auth-httplib2==0.0.2",
    "google-cloud-bigquery==0.25.0",
    "google-cloud-core==0.25.0",
    "google-cloud-dataflow==2.1.0",
    "google-cloud-pubsub==0.26.0",
    "google-gax==0.15.15",
    "googleapis-common-protos==1.5.2",
    "googledatastore==7.0.1",
    "grpc-google-iam-v1==0.11.3",
    "grpcio==1.6.0",
    "httplib2==0.9.2",
    "idna==2.6",
    "mock==2.0.0",
    "oauth2client==3.0.0",
    "pbr==3.1.1",
    "ply==3.8",
    "proto-google-cloud-datastore-v1==0.90.4",
    "proto-google-cloud-pubsub-v1==0.15.4",
    "protobuf==3.3.0",
    "pyasn1==0.3.5",
    "pyasn1-modules==0.1.4",
    "PyYAML==3.12",
    "requests==2.18.4",
    "rsa==3.4.2",
    "six==1.10.0",
    "urllib3==1.22",
]


data_files = [os.path.basename(x)
              for x in glob.glob("classification/data/*.csv")]

setuptools.setup(
    name='vessel_classification',
    version='1.0',
    author='Alex Wilson',
    author_email='alexwilson@google.com',
    package_data={
        'classification.data': data_files
    },
    packages=[
        'common',
        'classification',
        'classification.data',
        'classification.models',
        'classification.models.prod',
        'classification.models.dev',
        'pipeline',
        'pipeline.objects',
        'pipeline.options',
        'pipeline.schemas',
    ],
    install_requires=DEPENDENCIES + DATAFLOW_PINNED_DEPENDENCIES
)