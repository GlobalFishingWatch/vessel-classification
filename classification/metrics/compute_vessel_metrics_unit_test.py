import importlib.machinery
import sys
import types

import numpy as np


def _install_optional_dependency_stubs():
    if "tensorflow" not in sys.modules:
        tf_stub = types.ModuleType("tensorflow")
        tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        sys.modules["tensorflow"] = tf_stub

    if "yattag" not in sys.modules:
        yattag_stub = types.ModuleType("yattag")
        yattag_stub.__spec__ = importlib.machinery.ModuleSpec("yattag", loader=None)
        sys.modules["yattag"] = yattag_stub


_install_optional_dependency_stubs()

from . import compute_vessel_metrics as cvm  # noqa: E402


def test_weights_by_class_uses_label_weight_mapping():
    y_true = np.array(["a", "a", "b", "c"])
    y_pred = np.array(["a", "b", "b", "c"])
    label_weights = {"a": 2.0, "b": 1.0}

    weights = cvm.weights_by_class(label_weights, y_true, y_pred)

    assert np.allclose(weights, np.array([0.4, 0.4, 0.2, 0.0]))


def test_weights_by_class_handles_no_matching_labels():
    y_true = np.array(["x", "y"])
    y_pred = np.array(["x", "y"])
    label_weights = {"a": 2.0}

    weights = cvm.weights_by_class(label_weights, y_true, y_pred)

    assert np.allclose(weights, np.array([0.0, 0.0]))
