import setuptools
from classification import params

setuptools.setup(
    name='vessel_classification',
    version='1.0',
    author='Alex Wilson',
    author_email='alexwilson@google.com',
    package_data={
        'classification.data':
        [params.metadata_file, 'combined_fishing_ranges.csv']
    },
    packages=[
        'common', 'classification', 'classification.data',
        'classification.models', 'classification.models.alex',
        'classification.models.hernan', 'classification.models.tim'
    ],
    install_requires=[
        'NewlineJSON'
    ])
