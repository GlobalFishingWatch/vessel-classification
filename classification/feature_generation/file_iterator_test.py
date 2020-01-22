
from . import file_iterator
import posixpath as pp
import pytest

id_path = "gs://machine-learning-dev-ttl-120d/features/mmsi_features_v20191126/ids/part-00000-of-00001.txt"

def test_GCSFile():
    path = pp.join(id_path)
    with file_iterator.GCSFile(path) as fp:
        lines = fp.read().strip().split()
    assert lines[:2] == [b'900410135', b'413222478']


if __name__ == '__main__':
    tf.test.main()




