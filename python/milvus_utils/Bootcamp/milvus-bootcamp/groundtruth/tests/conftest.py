import struct
import numpy as np
import pytest
import torch


def has_gpu():
    return torch.cuda.device_count() > 0


@pytest.fixture
def skip_if_no_gpu():
    if not has_gpu():
        pytest.skip("No CUDA GPU available — skipping GPU-dependent tests")


def write_fbin(path, data: np.ndarray):
    n, d = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        data.astype(np.float32).tofile(f)


def write_fvecs(path, data: np.ndarray):
    with open(path, "wb") as f:
        for row in data:
            f.write(struct.pack("<i", row.shape[0]))
            row.astype(np.float32).tofile(f)
