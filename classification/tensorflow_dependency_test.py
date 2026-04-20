import importlib.util

import pytest


pytestmark = pytest.mark.tensorflow


def test_tensorflow_dependency_available():
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is not installed in this environment.")
