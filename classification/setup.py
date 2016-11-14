import setuptools
setuptools.setup(
    name='vessel_classification',
    version='1.0',
    author='Alex Wilson',
    author_email='alexwilson@google.com',
    package_data={
        'classification.data':
        ['net_training_20161111.csv', 'combined_fishing_ranges.csv']
    },
    packages=[
        'common', 'classification', 'classification.data',
        'classification.models', 'classification.models.alex',
        'classification.models.hernan', 'classification.models.tim'
    ],
    install_requires=[
        'NewlineJSON'
    ])
