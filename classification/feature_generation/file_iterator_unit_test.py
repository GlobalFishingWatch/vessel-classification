import os
import shutil

from .file_iterator import GCSFile


def test_gcsfile_downloads_and_cleans_up(monkeypatch, tmp_path):
    source_file = tmp_path / "part-00000-of-00001.txt"
    source_file.write_bytes(b"abc\n123\n")

    def fake_check_call(cmd):
        assert cmd[0] == "gsutil"
        assert cmd[1] == "cp"
        shutil.copyfile(source_file, cmd[3])

    monkeypatch.setattr("subprocess.check_call", fake_check_call)

    handle = GCSFile("gs://bucket/path/part-00000-of-00001.txt")
    with handle as fp:
        assert fp.read() == b"abc\n123\n"
        temp_dir = handle.temp_dir
        assert os.path.isdir(temp_dir)

    assert not os.path.exists(temp_dir)
