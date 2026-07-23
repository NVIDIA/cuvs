# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import vamana


def _gen_data(shape, dtype):
    rng = np.random.default_rng(12345)
    if dtype == np.float32:
        return rng.random(shape, dtype=np.float32)
    if dtype == np.float16:
        return rng.random(shape, dtype=np.float32).astype(np.float16)
    if dtype == np.int8:
        # keep small magnitude to avoid overflow if used elsewhere
        return rng.integers(low=-10, high=10, size=shape, dtype=np.int8)
    if dtype == np.uint8:
        return rng.integers(low=0, high=20, size=shape, dtype=np.uint8)
    raise AssertionError("unexpected dtype in test helper")


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
def test_vamana_build_basic(dtype):
    n_rows, n_cols = 1000, 16
    data = _gen_data((n_rows, n_cols), dtype)
    data_dev = device_ndarray(data)

    params = vamana.IndexParams(metric="sqeuclidean")
    idx = vamana.build(params, data_dev)

    # Basic sanity: object type and flags
    assert isinstance(idx, vamana.Index)
    assert idx.trained is True
    # repr should include trained flag
    assert "Index(type=Vamana" in repr(idx)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.skip(
    reason="Skipping host build test because of CUDA error "
    "in C++ API. Reference issue: "
    "https://github.com/rapidsai/cuvs/issues/1380"
)
def test_vamana_build_basic_host(dtype):
    n_rows, n_cols = 512, 12
    data = _gen_data((n_rows, n_cols), dtype)  # host array

    params = vamana.IndexParams(metric="sqeuclidean")
    idx = vamana.build(params, data)

    assert isinstance(idx, vamana.Index)
    assert idx.trained is True


@pytest.mark.parametrize("include_dataset", [True, False])
def test_vamana_serialize(tmp_path, include_dataset):
    n_rows, n_cols = 256, 8
    data = _gen_data((n_rows, n_cols), np.float32)
    data_dev = device_ndarray(data)

    params = vamana.IndexParams()
    idx = vamana.build(params, data_dev)

    out_path = tmp_path / "vamana_index.bin"
    vamana.save(str(out_path), idx, include_dataset=include_dataset)

    assert out_path.exists() and out_path.stat().st_size > 0


def test_vamana_build_rejects_unsupported_dtype():
    # e.g. float64 is not supported for Vamana build
    n_rows, n_cols = 64, 8
    data = np.random.default_rng(1).random((n_rows, n_cols), dtype=np.float64)
    from pylibraft.common import device_ndarray

    data_dev = device_ndarray(data)
    params = vamana.IndexParams()

    with pytest.raises(Exception):
        _ = vamana.build(params, data_dev)
