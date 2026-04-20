import importlib.util
import os
import shutil

def _has_tensorflow():
    return importlib.util.find_spec("tensorflow") is not None


def _enable_gcs_tests():
    return shutil.which("gsutil") is not None and os.environ.get("RUN_GCS_TESTS") == "1"


collect_ignore = []

if not _has_tensorflow():
    collect_ignore.extend([
        "classification/metadata_test.py",
        "classification/models/models_test.py",
        "classification/feature_generation/feature_generation_test.py",
        "classification/feature_generation/fishing_feature_generation_test.py",
        "classification/feature_generation/vessel_feature_generation_test.py",
    ])

if not _enable_gcs_tests():
    collect_ignore.extend([
        "classification/feature_generation/file_iterator_test.py",
        "classification/feature_generation/feature_generation_test.py",
        "classification/feature_generation/fishing_feature_generation_test.py",
        "classification/feature_generation/vessel_feature_generation_test.py",
    ])
