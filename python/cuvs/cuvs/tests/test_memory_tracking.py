# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import time

import cupy as cp

from cuvs.common import Resources


def test_memory_tracking_writes_csv(tmp_path):
    """Allocate a couple of small device buffers under a tracking
    Resources handle and confirm that the CSV reporter wrote at least
    one row before the handle was destroyed.
    """
    csv = tmp_path / "alloc.csv"

    res = Resources(
        memory_tracking_csv_path=str(csv),
        memory_tracking_sample_interval_ms=2,
    )
    try:
        a = cp.zeros((1024,), dtype=cp.float32)
        b = cp.zeros((2048,), dtype=cp.float32)
        res.sync()
        # Give the background reporter enough time to emit at least one
        # sample (interval is 2 ms above).
        time.sleep(0.05)
        del a, b
    finally:
        # Destroying the handle flushes the CSV and restores the
        # global host/device memory resources.
        del res

    assert csv.exists(), f"expected csv file at {csv}"
    assert csv.stat().st_size > 0, "tracking csv should be non-empty"

    lines = csv.read_text().splitlines()
    assert lines, "expected at least one line (header) in the csv"
