import os
import shutil

import pytest


pytestmark = pytest.mark.integration


def test_integration_dependencies_available():
    if shutil.which("gsutil") is None or os.environ.get("RUN_GCS_TESTS") != "1":
        pytest.skip("Integration fixtures unavailable (need gsutil and RUN_GCS_TESTS=1).")
