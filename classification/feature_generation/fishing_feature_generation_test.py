
import datetime
import gc
import posixpath as pp
import tensorflow as tf
import numpy as np
from . import feature_generation
from . import fishing_feature_generation
from . import feature_utilities
from ..models import fishing_detection
from .. import metadata as metedata_mod
import logging
import pytest

MAX_ITERS = 100

# TODO: copy the referenced file to somewhere permanent
prefix = b"gs://machine-learning-dev-ttl-120d/features/mmsi_features_fishing_testpy3/"
fishing_ranges = metedata_mod.read_fishing_ranges('classification/data/det_ranges_mmsi_v20200114.csv')
metadata = fishing_detection.Model.read_metadata([b'416853000', b'204248000'], 
            'classification/data/det_info_mmsi_v20200114.csv', fishing_ranges, 'Training') 
mdl = fishing_detection.Model(14, metadata, 'minimal')




def test_predict_input_fn():
    path1 = prefix + b"features/416853000.tfrecord"
    input_fn = fishing_feature_generation.predict_input_fn(
                        [path1], 
                        mdl.num_feature_dimensions + 1, 
                        mdl.window_max_points,
                        datetime.datetime(2015,1,1), 
                        datetime.datetime(2015,12,31),
                        mdl.window,
                        parallelism=1)
    iterator = input_fn.make_one_shot_iterator()
    next_element = iterator.get_next()
    values = []
    with tf.compat.v1.Session() as sess:
        for _ in range(MAX_ITERS):
            try:
                x = sess.run(next_element)
                assert x['id'] in (b'416853000')
                td = [datetime.datetime.utcfromtimestamp(y) for y in x['time_ranges']]
                values.append(x.copy())
            except tf.errors.OutOfRangeError:
                break
        else:
            raise RuntimeError('too many elements retrieved')
    x = values[0]
    assert sorted(x.keys()) == ['features', 'id', 'time_ranges', 'timestamps']
    assert x['id'] == b'416853000'
    for x in values[:10]:
        td = [datetime.datetime.utcfromtimestamp(y) for y in x['time_ranges']]
        print(x['id'], td)

def test_predict_input_fn_out_of_range():
    path1 = prefix + b"features/416853000.tfrecord"
    input_fn = fishing_feature_generation.predict_input_fn(
                        [path1], 
                        mdl.num_feature_dimensions + 1, 
                        mdl.window_max_points,
                        datetime.datetime(2010,1,1), 
                        datetime.datetime(2010,12,31),
                        mdl.window,
                        parallelism=1)
    iterator = input_fn.make_one_shot_iterator()
    next_element = iterator.get_next()
    values = []
    with tf.compat.v1.Session() as sess:
        for _ in range(MAX_ITERS):
            try:
                x = sess.run(next_element)
                assert x['id'] in (b'416853000')
                td = [datetime.datetime.utcfromtimestamp(y) for y in x['time_ranges']]
                values.append(x.copy())
            except tf.errors.OutOfRangeError:
                break
        else:
            raise RuntimeError('too many elements retrieved')
    assert len(values) == 0


def test_input_fn():
    path = prefix + b"features/204248000.tfrecord"
    input_fn = fishing_feature_generation.input_fn(
                        metadata,
                        [path], 
                        mdl.num_feature_dimensions + 1, 
                        mdl.max_window_duration_seconds,
                        mdl.window_max_points,
                        mdl.min_viable_timeslice_length,
                        parallelism=1)
    iterator = input_fn.make_one_shot_iterator()
    next_element = iterator.get_next()
    vals = []
    with tf.compat.v1.Session() as sess:
        for _ in range(3):
            x = sess.run(next_element)
            vals.append(x)
    assert len(vals) == 3
    (obj_0, obj_1) = vals[0]
    assert sorted(obj_0.keys()) == ['features', 'id', 'time_ranges', 'timestamps']
    assert obj_0['id'] == b'204248000'





if __name__ == '__main__':
    tf.test.main()




