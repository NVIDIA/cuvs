import pytest
import struct
import numpy as np

from groundtruth.main import _read_fbin_header, _parse_fbin_paths


def test_read_fbin_header_valid(tmp_path):
    data = np.zeros((10, 4), dtype=np.float32)
    p = tmp_path / "a.fbin"

    with open(p, "wb") as f:
        f.write(struct.pack("<ii", 10, 4))
        data.tofile(f)

    n, d = _read_fbin_header(str(p))
    assert (n, d) == (10, 4)


def test_read_fbin_header_size_mismatch(tmp_path):
    p = tmp_path / "bad.fbin"
    with open(p, "wb") as f:
        f.write(struct.pack("<ii", 10, 4))  # no data

    with pytest.raises(SystemExit):
        _read_fbin_header(str(p))


def test_parse_fbin_paths_valid(tmp_path):
    p = tmp_path / "a.fbin"
    with open(p, "wb") as f:
        f.write(b"\x00" * 8)

    paths = _parse_fbin_paths(str(p))
    assert paths == [str(p)]


def test_parse_fbin_paths_wrong_ext(tmp_path):
    p = tmp_path / "a.bin"
    p.write_text("x")

    with pytest.raises(SystemExit):
        _parse_fbin_paths(str(p))
