# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
from pylibraft.common import device_ndarray

from cuvs.common import Resources
from cuvs.neighbors import brute_force


def test_resources_with_external_stream():
    """A handle built from an integer stream handle, as in the Resources
    docstring, can be synchronized and used to run an algorithm.
    """
    stream = cp.cuda.Stream()
    res = Resources(stream=stream.ptr)
    res.sync()

    rng = np.random.default_rng(0)
    dataset = rng.random((64, 8), dtype=np.float32)
    dataset_device = device_ndarray(dataset)
    index = brute_force.build(dataset_device, resources=res)
    _, neighbors = brute_force.search(
        index, dataset_device, k=1, resources=res
    )
    res.sync()

    np.testing.assert_array_equal(
        neighbors.copy_to_host().ravel(), np.arange(dataset.shape[0])
    )
