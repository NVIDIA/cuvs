# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.common import Dataset, make_device_padded_dataset
from cuvs.tests.ann_utils import generate_data


def test_empty_dataset_properties():
    ds = Dataset()
    assert ds.memory_type is None
    assert ds.layout is None
    assert ds.is_owning is None
    assert ds.dtype is None


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("n_cols", [32, 50])
@pytest.mark.parametrize("from_host", [True, False])
def test_make_device_padded_dataset(dtype, n_cols, from_host):
    n_rows = 128
    data = generate_data((n_rows, n_cols), dtype)
    source = data if from_host else device_ndarray(data)

    ds = make_device_padded_dataset(source)
    assert isinstance(ds, Dataset)
    assert ds.layout == "padded"
    assert ds.memory_type == "device"
    assert ds.dtype is not None
    if from_host:
        assert ds.is_owning is True
    else:
        assert ds.is_owning in (True, False)


def test_make_device_padded_dataset_device_aligned_is_view():
    data = generate_data((64, 32), np.float32)
    source = device_ndarray(data)
    ds = make_device_padded_dataset(source)
    assert ds.layout == "padded"
    assert ds.memory_type == "device"
    assert ds.is_owning is False


def test_make_device_padded_dataset_device_unaligned_is_owning():
    data = generate_data((64, 50), np.float32)
    source = device_ndarray(data)
    ds = make_device_padded_dataset(source)
    assert ds.layout == "padded"
    assert ds.memory_type == "device"
    assert ds.is_owning is True


def test_make_device_padded_dataset_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="not supported"):
        make_device_padded_dataset(np.zeros((8, 8), dtype=np.float64))


def test_make_device_padded_dataset_rejects_non_contiguous():
    data = np.asfortranarray(generate_data((16, 16), np.float32))
    with pytest.raises(ValueError, match="Row major"):
        make_device_padded_dataset(data)
