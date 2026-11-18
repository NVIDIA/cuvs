# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""hnswlib-style filtered nearest-neighbor search workflow.

The upstream hnswlib example uses a Python callback to restrict candidates.
This fixture keeps the same even-id filter contract with a small NumPy baseline
so the benchmark does not depend on compiling hnswlib in the task environment.
"""

from __future__ import annotations

import numpy as np


def make_data(dim: int = 16, num_elements: int = 10_000) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return np.float32(rng.random((num_elements, dim)))


def run_filtered_search(k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    data = make_data()
    num_elements = data.shape[0]
    even_ids = np.arange(0, num_elements, 2, dtype=np.int64)
    even_data = data[even_ids]

    if k < 1:
        raise ValueError("k must be positive")
    if k > len(even_ids):
        raise ValueError(f"k={k} exceeds the number of even-id candidates")

    labels = np.empty((num_elements, k), dtype=np.int64)
    distances = np.empty((num_elements, k), dtype=np.float32)
    for start in range(0, num_elements, 512):
        stop = min(start + 512, num_elements)
        diff = data[start:stop, None, :] - even_data[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        sorted_part = np.take_along_axis(part, order, axis=1)
        labels[start:stop] = even_ids[sorted_part]
        distances[start:stop] = np.take_along_axis(part_dist, order, axis=1)
    return labels, distances


def main() -> None:
    labels, distances = run_filtered_search(k=1)
    assert labels.shape == distances.shape == (10_000, 1)
    assert np.max(np.mod(labels, 2)) == 0
    even_self_recall = np.mean(labels.reshape(-1) == np.arange(len(labels)))
    print(f"even_only={bool(np.max(np.mod(labels, 2)) == 0)}")
    print(f"even_self_recall={even_self_recall:.3f}")


if __name__ == "__main__":
    main()
