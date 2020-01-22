import datetime
import logging
import numpy as np
import tempfile
import subprocess
import os
import resource
import shutil
import time

import tensorflow as tf

from .feature_utilities import np_array_extract_all_fixed_slices
from .feature_utilities import np_array_extract_slices_for_time_ranges
from .feature_utilities import np_pad_repeat_slice


class GCSFile(object):

    def __init__(self, path):
        self.gcs_path = path

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(self.temp_dir, os.path.basename(self.gcs_path))
        subprocess.check_call(['gsutil', 'cp', self.gcs_path, local_path])
        return self._process(local_path)

    def _process(self, path):
        return open(path, 'rb')

    def __exit__(self, *args):
        shutil.rmtree(self.temp_dir)






