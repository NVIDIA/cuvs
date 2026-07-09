# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cupy as cp

from cuvs.common import Resources


def test_resources_syncs_cupy_stream_pointer():
    # gh-issue: 1836 should not segfault when syncing a stream pointer from cupy
    stream = cp.cuda.Stream()
    resources = Resources(stream=stream.ptr)

    resources.sync()
