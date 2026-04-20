import pytest

from .gcp_config import GcpConfig


def test_make_from_env_name_invalid_environment_exits():
    with pytest.raises(SystemExit):
        GcpConfig.make_from_env_name("invalid", "job-1")
