import setuptools
import glob
import os

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
        'common', 'classification', 'classification.data',
        'classification.models', 'classification.models.prod',
        'classification.models.dev',
        'classification.models.dev.alex',
        'classification.models.dev.tim'
    ],
    install_requires=[
        'NewlineJSON'
    ])
