import tempfile
import subprocess
import os
import shutil


class GCSFile(object):

    def __init__(self, path):
        self.gcs_path = path

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(self.temp_dir, os.path.basename(self.gcs_path))
        subprocess.check_call(['gsutil', 'cp', self.gcs_path, local_path])
        self._file_handle = self._process(local_path)
        return self._file_handle

    def _process(self, path):
        return open(path, 'rb')

    def __exit__(self, *args):
        self._file_handle.close()
        shutil.rmtree(self.temp_dir)






