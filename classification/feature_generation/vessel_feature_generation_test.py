
import gc
import posixpath as pp
import tensorflow as tf
import numpy as np
from . import feature_generation
from . import vessel_feature_generation
from . import feature_utilities
from ..models import vessel_characterization
import logging
import pytest

# TODO: copy the referenced file to somewhere permanent
prefix = b"gs://machine-learning-dev-ttl-120d/features/mmsi_features_fishing_testpy3/"
metadata = vessel_characterization.Model.read_metadata([b'416853000'], 
            'classification/data/char_info_mmsi_v20200114.csv', {}, '0') 
mdl = vessel_characterization.Model(14, metadata, 'minimal')


def test_input_fn():
    path = prefix + b"features/416853000.tfrecord"
    input_fn = vessel_feature_generation.input_fn(
                        metadata,
                        [path], 
                        mdl.num_feature_dimensions + 1, 
                        mdl.max_window_duration_seconds,
                        mdl.window_max_points,
                        mdl.min_viable_timeslice_length,
                        objectives=mdl.training_objectives,
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
    assert sorted(obj_1.keys()) == ['Vessel-Crew-Size', 'Vessel-class', 'Vessel-engine-Power', 
                                    'Vessel-length', 'Vessel-tonnage']
    assert [np.argmax(obj_b['Vessel-class']) for (obj_a, obj_b) in vals] == [31] * 3





def test_predict_input_fn():
    path = prefix + b"features/416853000.tfrecord"
    input_fn = vessel_feature_generation.predict_input_fn(
                        [path], 
                        mdl.num_feature_dimensions + 1, 
                        [(1190549800.0, 1567037800.0)],
                        mdl.window_max_points,
                        mdl.min_viable_timeslice_length,
                        parallelism=1)
    iterator = input_fn.make_one_shot_iterator()
    next_element = iterator.get_next()
    values = []
    with tf.compat.v1.Session() as sess:
        while True:
            try:
                x = sess.run(next_element)
                values.append(x)
            except tf.errors.OutOfRangeError:
                break
    [x] = values
    assert sorted(x.keys()) == ['features', 'id', 'time_ranges', 'timestamps']


if __name__ == '__main__':
    tf.test.main()




