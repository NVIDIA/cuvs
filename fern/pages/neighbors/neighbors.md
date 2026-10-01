---
slug: user-guide/api-guides/indexing-guide
---

# Indexing Guide

Use these guides for NVIDIA cuVS nearest-neighbor indexing APIs, from exact search baselines to GPU-accelerated approximate indexes.

- [All-neighbors](/user-guide/api-guides/indexing-guide/all-neighbors): compute all-neighbors graph structures.
- [Brute-force](/user-guide/api-guides/indexing-guide/brute-force): compare every query against every vector for exact nearest-neighbor search.
- [CAGRA](/user-guide/api-guides/indexing-guide/cagra): build and search GPU-optimized graph indexes.
- [IVF-Flat](/user-guide/api-guides/indexing-guide/ivf-flat): partition vectors into inverted-file lists while storing full-precision vectors.
- [IVF-PQ](/user-guide/api-guides/indexing-guide/ivf-pq): combine inverted-file partitioning with product quantization for compact indexes.
- [Multi-GPU](/user-guide/api-guides/indexing-guide/multi-gpu): distribute supported nearest-neighbor indexes across multiple GPUs.
- [NN-Descent](/user-guide/api-guides/indexing-guide/nn-descent): build approximate nearest-neighbor graphs with an iterative algorithm.
- [ScaNN](/user-guide/api-guides/indexing-guide/sca-nn): combine partitioning, quantization, and refinement for high-quality approximate search.
- [Vamana](/user-guide/api-guides/indexing-guide/vamana): build graph indexes designed for large-scale and disk-backed search workflows.
