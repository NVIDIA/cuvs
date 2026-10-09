import numpy as np
import pytest

from groundtruth.main import stream_fbin_batches
from groundtruth.tests.conftest import write_fbin


def test_stream_single_file_batches(tmp_path):
    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    p = tmp_path / "a.fbin"
    write_fbin(p, data)

    batches = list(
        stream_fbin_batches(
            [str(p)],
            batch_size=4,
            dim=2,
        )
    )

    offsets = [o for o, _ in batches]
    sizes = [b.shape[0] for _, b in batches]

    assert offsets == [0, 4, 8]
    assert sizes == [4, 4, 2]


def test_stream_multiple_files_offsets(tmp_path):
    d1 = np.zeros((3, 2), np.float32)
    d2 = np.ones((2, 2), np.float32)

    p1 = tmp_path / "a.fbin"
    p2 = tmp_path / "b.fbin"
    write_fbin(p1, d1)
    write_fbin(p2, d2)

    batches = list(
        stream_fbin_batches(
            [str(p1), str(p2)],
            batch_size=10,
            dim=2,
        )
    )

    assert batches[0][0] == 0
    assert batches[1][0] == 3


def test_stream_invalid_batch_size(tmp_path):
    p = tmp_path / "a.fbin"
    write_fbin(p, np.zeros((1, 1), np.float32))

    with pytest.raises(SystemExit):
        list(
            stream_fbin_batches(
                [str(p)],
                batch_size=0,
                dim=1,
            )
        )
