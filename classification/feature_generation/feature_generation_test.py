
import gc
import posixpath as pp
import tensorflow as tf
import numpy as np
from . import feature_generation
from ..models import vessel_characterization
import pytest


metadata = vessel_characterization.Model.read_metadata([b'416853000', b'100209703', b'204225000'], 
            'classification/data/char_info_mmsi_v20200114.csv', {}, '0') 
# TODO: copy the referenced file to somewhere permanent
prefix = b"gs://machine-learning-dev-ttl-120d/features/mmsi_features_fishing_testpy3/"
                  

def test_read_input_fn_one_shot():
    paths = ([prefix + b"features/416853000.tfrecord"]# + 
             # [prefix + b"features/205285000.tfrecord"] + 
             # [prefix + b"features/204225000.tfrecord"]
             )
    dataset = feature_generation.read_input_fn_one_shot(paths, 15)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    values = []
    with tf.Session() as sess:
        while True:
            try:
                id_, data = sess.run(next_element)
                values.append((id_, data))
            except tf.errors.OutOfRangeError:
                break
    assert len(values) == len(paths)
    [(id_, data)] = [x for x in values if x[0] == b'416853000']
    assert id_ == b'416853000'
    assert data.shape == (11845, 15), data.shape



def test_read_input_fn_infinite():
    path = prefix + b"features/416853000.tfrecord"
    dataset = feature_generation.read_input_fn_infinite([path], 15)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    values = []
    with tf.Session() as sess:
        for _ in range(3):
            id_, data = sess.run(next_element)
            values.append((id_, data))
    assert len(values) == 3
    id_, data = values[0]
    print(metadata.id_map_int2bytes)
    assert metadata.id_map_int2bytes[id_] == b'416853000'
    assert data.shape == (11845, 15), data.shape
    # assert np.allclose(data[0], [ 1.4905498e+09, 9.3955746e+00,  8.2576685e+00,  2.6900861e-01,
    #                       2.7795976e-01, -6.9944441e-01,  9.0191650e-01,  4.3191043e-01,
    #                       6.1232343e-17,  1.0000000e+00, -8.6529666e-01,  2.8903718e+00,
    #                       0.0000000e+00,  0.0000000e+00,  0.0000000e+00])

if __name__ == '__main__':
    tf.test.main()




