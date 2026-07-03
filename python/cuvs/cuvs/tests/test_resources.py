# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for cuVS resources and pylibraft handle compatibility."""

import numpy as np
import pytest
from pylibraft.common import DeviceResources, device_ndarray

from cuvs.common import Resources
from cuvs.neighbors import brute_force


def test_resources_from_pylibraft_handle():
    """Test creating Resources from a pylibraft DeviceResources handle."""
    pylibraft_handle = DeviceResources()

    # Create cuVS Resources from pylibraft handle
    resources = Resources(handle=pylibraft_handle.getHandle())

    # Should have get_c_obj method that returns the same pointer
    assert hasattr(resources, "get_c_obj")
    assert hasattr(resources, "sync")
    assert resources.get_c_obj() == pylibraft_handle.getHandle()


def test_auto_sync_rejects_invalid_object():
    """Test that auto_sync_resources raises TypeError for invalid objects."""
    n_samples, n_features = 10, 4
    dataset = np.random.random((n_samples, n_features)).astype(np.float32)
    dataset_device = device_ndarray(dataset)

    with pytest.raises(TypeError, match="cuVS Resources or pylibraft"):
        brute_force.build(dataset_device, resources="invalid")

    with pytest.raises(TypeError, match="cuVS Resources or pylibraft"):
        brute_force.build(dataset_device, resources=123)


def test_brute_force_build_with_pylibraft_handle():
    """Test brute_force.build with pylibraft DeviceResources."""
    n_samples, n_features = 100, 10

    dataset = np.random.random((n_samples, n_features)).astype(np.float32)
    dataset_device = device_ndarray(dataset)

    # Use pylibraft DeviceResources instead of cuVS Resources
    handle = DeviceResources()

    index = brute_force.build(dataset_device, resources=handle)

    # Explicit sync since we passed our own handle
    handle.sync()

    assert index.trained


def test_multi_gpu_resources_from_pylibraft_snmg_handle():
    """Test creating MultiGpuResources from a pylibraft DeviceResourcesSNMG."""
    from pylibraft.common import DeviceResourcesSNMG
    from cuvs.common import MultiGpuResources

    pylibraft_handle = DeviceResourcesSNMG()

    # Create cuVS MultiGpuResources from pylibraft handle
    resources = MultiGpuResources(handle=pylibraft_handle.getHandle())

    assert hasattr(resources, "get_c_obj")
    assert hasattr(resources, "sync")
    assert resources.get_c_obj() == pylibraft_handle.getHandle()


def test_mg_algorithm_with_pylibraft_snmg_handle():
    """Test multi-GPU algorithm with pylibraft DeviceResourcesSNMG."""
    from pylibraft.common import DeviceResourcesSNMG

    from cuvs.neighbors.mg import cagra as mg_cagra

    n_rows, n_cols = 1000, 8

    dataset = np.random.random((n_rows, n_cols)).astype(np.float32)

    # Use pylibraft DeviceResourcesSNMG
    handle = DeviceResourcesSNMG()

    build_params = mg_cagra.IndexParams(
        graph_degree=16, intermediate_graph_degree=32
    )
    index = mg_cagra.build(build_params, dataset, resources=handle)

    # Explicit sync since we passed our own handle
    handle.sync()

    assert index.trained
