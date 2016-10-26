import setuptools
setuptools.setup(
    name='vessel_classification',
    version='1.0',
    author='Alex Wilson',
    author_email='alexwilson@google.com',
    install_requires=[
        "newlinejson",
    ],
    package_data={
        'classification.data':
        ['combined_classification_list.csv', 'combined_fishing_ranges.csv']
    },
    packages=[
        'common', 'classification', 'classification.data',
        'classification.models', 'classification.models.alex',
        'classification.models.hernan', 'classification.models.tim'
    ],
    install_requires=['NewlineJSON'])
