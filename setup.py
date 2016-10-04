import setuptools
setuptools.setup(
    name='vessel_classification',
    version='1.0',
    author='Alex Wilson',
    author_email='alexwilson@google.com',
    data_files=[('data', ['classification/data/combined_classification_list.csv'])],
    packages=['classification', 'classification.models',
              'classification.models.alex', 'classification.models.tim'])
