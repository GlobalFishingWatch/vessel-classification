import datetime
from collections import namedtuple

import numpy as np

from . import feature_utilities as fu


def test_np_pad_repeat_slice_repeats_to_target_length():
    source = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    padded = fu.np_pad_repeat_slice(source, window_size=5)

    assert padded.shape == (5, 2)
    assert np.allclose(padded[:2], source)
    assert np.allclose(padded[2:4], source)
    assert np.allclose(padded[4], source[0])


def test_np_zero_pad_slice_returns_window_size_slice():
    source = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    result = fu.np_zero_pad_slice(
        source, window_size=5, random_state=np.random.RandomState(0)
    )

    assert result.shape == (5, 2)


def test_np_array_random_fixed_length_extract_output_shape():
    source = np.arange(24, dtype=np.float32).reshape(12, 2)
    result = fu.np_array_random_fixed_length_extract(
        random_state=np.random.RandomState(0),
        input_series=source,
        output_length=5,
    )

    assert result.shape == (5, 2)


def test_cook_features_splits_timestamps_and_features():
    series = np.array(
        [[1000.0, 1.5, 2.5], [1005.0, 3.5, 4.5], [1010.0, 5.5, 6.5]],
        dtype=np.float32,
    )
    features, timestamps, time_range, id_ = fu.cook_features(series, b"v-1")

    assert features.shape == (1, 3, 2)
    assert timestamps.tolist() == [1000, 1005, 1010]
    assert time_range.tolist() == [1000, 1010]
    assert id_ == b"v-1"


def test_extract_n_random_fixed_times_returns_expected_shapes():
    timestamps = np.arange(1_000, 1_010, dtype=np.float32)
    other = np.arange(10, dtype=np.float32)
    series = np.stack([timestamps, other], axis=1)

    features, ts, ranges, ids = fu.extract_n_random_fixed_times(
        random_state=np.random.RandomState(1),
        input_series=series,
        n=3,
        max_time_delta=5,
        output_length=4,
        id_=b"boat-7",
        min_timeslice_size=3,
    )

    assert features.shape == (3, 4, 1)
    assert ts.shape == (3, 4)
    assert ranges.shape == (3, 2)
    assert ids.tolist() == [b"boat-7", b"boat-7", b"boat-7"]


def test_extract_n_random_fixed_points_returns_data():
    FishingRange = namedtuple("FishingRange", ["start_time", "end_time", "is_fishing"])
    base = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    timestamps = np.arange(1_577_836_800, 1_577_836_810, dtype=np.float32)
    feat = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    series = np.stack([timestamps, feat], axis=1)
    ranges = [FishingRange(base, base + datetime.timedelta(seconds=5), 1.0)]

    features, ts, time_ranges, ids = fu.extract_n_random_fixed_points(
        random_state=np.random.RandomState(2),
        input_series=series,
        n=2,
        output_length=4,
        id_=b"id-10",
        selection_ranges=ranges,
    )

    assert len(features) == 2
    assert len(ts) == 2
    assert len(time_ranges) == 2
    assert list(ids) == [b"id-10", b"id-10"]


def test_extract_slices_for_time_ranges_uses_random_offset():
    class DummyRandomState(object):
        def randint(self, low, high=None):
            return 0

    timestamps = np.arange(1000, 1020, dtype=np.float32)
    features = np.arange(20, dtype=np.float32)
    series = np.stack([timestamps, features], axis=1)
    time_ranges = [(1002, 1018)]

    result = fu.np_array_extract_slices_for_time_ranges(
        random_state=DummyRandomState(),
        input_series=series,
        id_=b"a",
        time_ranges=time_ranges,
        window_size=5,
        min_points_for_classification=3,
    )
    timestamps_out = result[1]
    assert timestamps_out.shape == (1, 5)
    assert timestamps_out.tolist() == [[1002, 1003, 1004, 1005, 1006]]
