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
from pathlib import Path

package = __import__('classification')

DEPENDENCIES = [
    "google-api-python-client",
    "six>=1.13.0"
]


data_files = [
    os.path.basename(x)
    for pattern in ("classification/data/*.csv", "classification/data/*.txt")
    for x in glob.glob(pattern)
]

readme = Path("README.md").read_text(encoding="utf-8")

setuptools.setup(
    name='vessel_inference',
    version=package.__version__,
    author=package.__author__,
    author_email=package.__email__,
    description=package.__doc__.strip(),
    long_description=readme,
    long_description_content_type="text/markdown",
    package_data={
        'classification.data': data_files
    },
    packages=setuptools.find_packages(include=["classification*", "common*"]),
    install_requires=DEPENDENCIES,
    extras_require={
        "dev": ["pytest>=7.0"]
    },
    python_requires=">=3.8",
)

