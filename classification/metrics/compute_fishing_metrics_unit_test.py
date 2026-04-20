import csv
import importlib.machinery
import sys
import types
from pathlib import Path

import pytz


def _install_optional_dependency_stubs():
    if "yattag" not in sys.modules:
        yattag_stub = types.ModuleType("yattag")
        yattag_stub.__spec__ = importlib.machinery.ModuleSpec("yattag", loader=None)
        sys.modules["yattag"] = yattag_stub

    if "pandas_gbq" not in sys.modules:
        pandas_gbq_stub = types.ModuleType("pandas_gbq")
        pandas_gbq_stub.__spec__ = importlib.machinery.ModuleSpec("pandas_gbq", loader=None)
        pandas_gbq_stub.gbq = types.SimpleNamespace(GenericGBQException=Exception)
        sys.modules["pandas_gbq"] = pandas_gbq_stub

    if "tensorflow" not in sys.modules:
        tf_stub = types.ModuleType("tensorflow")
        tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        tf_stub.convert_to_tensor = lambda x: x
        tf_stub.gather = lambda table, index: table[index]
        sys.modules["tensorflow"] = tf_stub


_install_optional_dependency_stubs()

from . import compute_fishing_metrics as cfm  # noqa: E402


def _write_ranges_csv(path: Path, rows):
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "start_time", "end_time", "is_fishing"])
        writer.writerows(rows)


def test_compare_fishing_localisation_handles_empty_overlap(tmp_path):
    ranges_file = tmp_path / "ranges.csv"
    _write_ranges_csv(
        ranges_file,
        [["v1", "2019-01-01T00:00:00Z", "2019-01-01T00:01:00Z", "1"]],
    )
    inferred_ranges = {"v2": []}

    result = cfm.compare_fishing_localisation(
        inferred_ranges=inferred_ranges,
        fishing_range_path=str(ranges_file),
        label_map={"v1": "trawlers"},
        split_map={"v1": "0"},
        split=0,
    )

    assert result.true_fishing_by_id == {}
    assert result.pred_fishing_by_id == {}


def test_compare_fishing_localisation_produces_masked_vectors(tmp_path):
    ranges_file = tmp_path / "ranges.csv"
    _write_ranges_csv(
        ranges_file,
        [["v1", "2019-01-01T00:00:00Z", "2019-01-01T00:02:00Z", "1"]],
    )

    inferred_ranges = {
        "v1": [
            cfm.FishingRange(
                True,
                cfm.dateutil.parser.parse("2019-01-01T00:00:00Z").replace(tzinfo=pytz.utc),
                cfm.dateutil.parser.parse("2019-01-01T00:02:00Z").replace(tzinfo=pytz.utc),
            )
        ]
    }

    result = cfm.compare_fishing_localisation(
        inferred_ranges=inferred_ranges,
        fishing_range_path=str(ranges_file),
        label_map={"v1": "trawlers"},
        split_map={"v1": "0"},
        split=0,
    )

    assert "v1" in result.true_fishing_by_id
    assert "v1" in result.pred_fishing_by_id
    assert len(result.true_fishing_by_id["v1"]) == 3
    assert len(result.pred_fishing_by_id["v1"]) == 3
