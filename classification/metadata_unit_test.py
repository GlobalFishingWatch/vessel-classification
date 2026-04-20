import csv
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np


if "tensorflow" not in sys.modules:
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.convert_to_tensor = lambda x: x
    tf_stub.gather = lambda table, index: table[index]
    tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    sys.modules["tensorflow"] = tf_stub

from . import metadata


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def test_read_vessel_multiclass_metadata_lines_from_fixture():
    metadata_file = FIXTURES_DIR / "metadata_sample.csv"
    with metadata_file.open("r", newline="") as fp:
        parsed_lines = csv.DictReader(fp)
        available_ids = {b"100001", b"100002", b"100003"}
        result = metadata.read_vessel_multiclass_metadata_lines(
            available_ids, parsed_lines, {}
        )

    assert set(result.metadata_by_split.keys()) == {"Training", "Test"}
    assert set(result.metadata_by_split["Training"].keys()) == {b"100002", b"100003"}
    assert set(result.metadata_by_split["Test"].keys()) == {b"100001"}
    assert result.vessel_label("label", b"100003") == "tug|trawlers"


def test_read_fishing_ranges_from_fixture():
    ranges_file = FIXTURES_DIR / "fishing_ranges_sample.csv"
    ranges = metadata.read_fishing_ranges(str(ranges_file))

    assert set(ranges.keys()) == {b"100002", b"100003"}
    assert len(ranges[b"100002"]) == 1
    assert ranges[b"100002"][0].is_fishing == 1.0


def test_weighted_training_list_respects_boundary():
    rows = {
        "Training": {
            b"100002": ({"label": "trawlers"}, 1.0),
            b"100003": ({"label": "tug"}, 1.0),
        },
        "Test": {},
    }
    vessel_metadata = metadata.VesselMetadata(rows, {})
    values = vessel_metadata.weighted_training_list(
        random_state=np.random.RandomState(0),
        split="Training",
        max_replication_factor=2,
        boundary=4,
    )

    assert len(values) == 4
    assert set(values).issubset({b"100002", b"100003"})


def test_stable_hash_is_deterministic():
    assert metadata.stable_hash("abc123") == metadata.stable_hash("abc123")


def test_unknown_vessel_label_does_not_raise():
    lines = [
        {"id": "100050", "label": "unknown_label", "split": "Training", "idhash": "1"},
        {"id": "100051", "label": "trawlers", "split": "Training", "idhash": "2"},
    ]
    available_ids = {b"100050", b"100051"}

    result = metadata.read_vessel_multiclass_metadata_lines(available_ids, lines, {})

    assert b"100051" in result.metadata_by_split["Training"]
