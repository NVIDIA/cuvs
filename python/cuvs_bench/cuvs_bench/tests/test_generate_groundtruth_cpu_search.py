#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for generate_groundtruth CPU search metrics."""

import numpy as np
import pytest

from cuvs_bench.generate_groundtruth.__main__ import cpu_search


class TestCpuSearchMetricName:
    """Regression for NVIDIA/cuvs#2036: cpu_search must accept sqeuclidean."""

    def test_sqeuclidean_default_path(self):
        dataset = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
        )
        queries = np.array([[0.0, 0.0]], dtype=np.float32)

        distances, indices = cpu_search(
            dataset, queries, k=2, metric="sqeuclidean"
        )

        assert indices.shape == (1, 2)
        assert indices[0, 0] == 0
        assert distances[0, 0] == pytest.approx(0.0)

    def test_default_metric_is_sqeuclidean(self):
        dataset = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        queries = np.array([[0.0, 0.0]], dtype=np.float32)

        distances, indices = cpu_search(dataset, queries, k=1)

        assert indices[0, 0] == 0
        assert distances[0, 0] == pytest.approx(0.0)
