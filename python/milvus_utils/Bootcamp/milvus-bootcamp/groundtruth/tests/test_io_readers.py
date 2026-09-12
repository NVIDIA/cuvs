import numpy as np
from groundtruth.main import read_fbin, read_fvecs



def test_read_fbin(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    p = tmp_path / "x.fbin"

    with open(p, "wb") as f:
        f.write(np.array([3, 4], dtype=np.int32).tobytes())
        data.tofile(f)

    out = read_fbin(str(p))
    assert out.shape == (3, 4)
    assert np.allclose(out, data)


def test_read_fvecs(tmp_path):
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    p = tmp_path / "x.fvecs"

    with open(p, "wb") as f:
        for row in data:
            f.write(np.array([2], dtype=np.int32).tobytes())
            row.tofile(f)

    out = read_fvecs(str(p))
    assert out.shape == (2, 2)
    assert np.allclose(out, data)
