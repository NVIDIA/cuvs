---
name: accelerated-computing-cuvs
description: Official NVIDIA-authored guidance for NVIDIA cuVS GPU vector search, approximate nearest neighbors, CAGRA, IVF-Flat, IVF-PQ, HNSW export, embedding retrieval, semantic search, vector database backends, RAG search, GPU clustering.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - cuvs
    - vector-search
    - ann
    - rag
    - retrieval
---

# cuVS Implementer's Guide

## Compatibility

- Release tracked by this skill: 26.06. Verify platform support with the RAPIDS install selector.
- Requires an NVIDIA GPU and a matching RAPIDS/cuVS install.
- When the installed version or docs differ, use `references/version-boundaries.md`.

## Naming

Use NVIDIA library-first wording in user-facing answers. Keep literal RAPIDS/rapidsai URLs, package names, and release metadata when citing sources.

## Role

You are a cuVS expert helping an implementer build GPU-accelerated vector search pipelines, or helping a database/search-platform integrator choose how cuVS fits behind their service. The user understands their embedding data and search requirements — your job is to select the right index or integration pattern, configure it correctly, and wire it into their pipeline. Vector search quality vs speed is a tradeoff that requires deliberate configuration.

## Critical Rules

1. **Choose the index for the workload.** CAGRA for high-recall search that fits on the GPU. IVF-Flat when full-precision lists are enough. Scalar quantization when bandwidth is the limit and the recall loss can stay small. IVF-PQ when index size is the limit. See `references/index-selection.md`.
2. **Handle dtype deliberately.** `float32` is the safe baseline for dense embeddings and exact ground-truth paths. Preserve `float16`, `int8`, or `uint8` when the chosen API accepts them and the data is already in that dtype or memory-bound, then validate recall. The per-API table is in `references/version-boundaries.md`.
3. **Three dimensions to balance:** build time, search quality (recall), search throughput. You cannot maximize all three simultaneously.
4. **Tune search before build.** CAGRA: `itopk_size`, then `graph_degree`, then `intermediate_graph_degree`. IVF: about 1,000–10,000 vectors per list, then `n_probes`, then compression, then `refine`. Compare latency, throughput, build time, and memory inside one recall bucket: 80–89%, 90–94%, 95–98%, or 99%+. For a hash-sharded database, tune one production segment, not the full corpus. See `references/index-selection.md`.
5. **Match exactness before speed.** Use cuVS brute force for exact top-k,
   exact filtered search, callback-filter parity, and recall ground truth. Use
   ANN indexes when approximate recall is acceptable and measured.
6. **Place the GPU where the workload benefits.** Under about 100K vectors, start with brute force or CPU HNSW unless dimensions or query batches are large. Use IVF-Flat, scalar quantization, or IVF-PQ when a large dataset needs fast ingest. Use CAGRA when a high-recall index fits in GPU memory. Use a GPU build with CPU HNSW serving when indexing is the bottleneck and search stays on CPU. A corpus that does not fit in memory is a DiskANN problem, not an in-memory cuVS index. When the production path is GPU search, report CPU timing separately.
7. **Confirm the quantizer before importing it.** Use an IVF-SQ index only when the installed package exposes one; otherwise use `cuvs.preprocessing.quantize.scalar`. Recover recall with larger `n_probes` or `itopk_size`, then overfetch and `refine`.
8. **Keep cuVS inputs device-resident.** Prefer CuPy, PyTorch CUDA tensors, or other `__cuda_array_interface__` arrays. NumPy host arrays require a visible host→device transfer step before build/search.
9. **Apply filters before search.** For filtered vector search, use cuVS
   pre-search filters (`filters.from_bitset` or `filters.from_bitmap`) so the
   search ranks only eligible vectors. Exact filtered top-k uses
   `brute_force.search(..., prefilter=...)`; ANN filtered search uses
   `search(..., filter=...)` and needs recall validation against the exact path.
10. **Stay at the vector-search layer.** For embedding search services, use cuVS indexes. Use RAFT for lower-level primitives and custom library work when the task is below the index/service layer.
11. **Use the C API for binary-stable product integrations.** For vector databases, search engines, language bindings, or packaged apps that need ABI stability, choose the cuVS C API boundary and check the installed ABI window. See `references/ann-patterns.md` and `references/version-boundaries.md`.

## Search Shape

Pick the CAGRA search kernel from the query count before tuning `itopk_size`.

```python
from cuvs.neighbors import cagra

# Fewer than about 10 queries. algo="auto" also selects from the query count.
search_params = cagra.SearchParams(itopk_size=64, algo="multi_cta")

# Larger batches. persistent applies only to single_cta.
search_params = cagra.SearchParams(
    itopk_size=64,
    algo="single_cta",
    persistent=True,
)
```

A Python loop that searches one vector and copies each result to the host measures the round trip, not GPU search.

## Core Workflow

```python
import cupy as cp
from cuvs.neighbors import cagra

# 1. Prepare data — shape (n_vectors, n_dims); float32 is a safe dense default
dataset = cp.random.random((1_000_000, 128), dtype=cp.float32)

# 2. Build index
index_params = cagra.IndexParams(
    metric="sqeuclidean",   # or "cosine" when IndexParams accepts it
    intermediate_graph_degree=128,
    graph_degree=64,
)
index = cagra.build(index_params, dataset)

# 3. Search
queries = cp.random.random((1000, 128), dtype=cp.float32)
search_params = cagra.SearchParams(itopk_size=64)   # higher = better recall, slower
distances, neighbors = cagra.search(search_params, index, queries, k=10)

# neighbors: shape (n_queries, k) — indices into dataset
# distances: shape (n_queries, k) — distance values
```

IVF-Flat, IVF-PQ, and `refine` are in `references/index-selection.md`.

## Interoperability

```python
import numpy as np
import torch

# From NumPy: make the host→device transfer explicit
np_data = np.random.random((100_000, 128)).astype(np.float32)
cp_data = cp.asarray(np_data)
index = cagra.build(cagra.IndexParams(), cp_data)

# From PyTorch GPU tensor
torch_data = torch.randn(100_000, 128, device="cuda", dtype=torch.float32)
index = cagra.build(cagra.IndexParams(), torch_data)

# Output to CuPy or PyTorch
distances, neighbors = cagra.search(search_params, index, queries, k=10)
cupy_neighbors = cp.asarray(neighbors)
torch_neighbors = torch.as_tensor(neighbors, device="cuda")
```

## Cosine Similarity

CAGRA and IVF-Flat accept `metric="cosine"` when the installed `IndexParams` documents it. Do the IVF-PQ check in `references/version-boundaries.md` before using `cosine` there.

```python
index_params = cagra.IndexParams(metric="cosine")
index = cagra.build(index_params, dataset)
```

When the installed version rejects `cosine`, L2-normalize the dataset and the queries, then use `metric="inner_product"`.

```python
norms = cp.linalg.norm(dataset, axis=1, keepdims=True)
normalized = dataset / norms
index_params = cagra.IndexParams(metric="inner_product")
index = cagra.build(index_params, normalized)
```

## Recall Validation

Measure recall against brute-force ground truth on a representative sample before production. Recall@k is the number of approximate neighbors that also appear in that query's exact top-k, summed over queries and divided by `n_queries * k`. Use the same metric for the ground-truth index and the approximate index.

```python
from cuvs.neighbors import brute_force

k = 10
gt_index = brute_force.build(dataset, metric="sqeuclidean")
_, gt_neighbors = brute_force.search(gt_index, queries, k=k)
_, approx_neighbors = cagra.search(search_params, index, queries, k=k)

approx = cp.asarray(approx_neighbors)
truth = cp.asarray(gt_neighbors)
matches = (approx[:, :, None] == truth[:, None, :]).any(axis=2).sum(axis=1)
recall = (matches.sum() / (approx.shape[0] * k)).item()
print(f"Recall@{k}: {recall:.3f}")
```

## Benchmarking

Run cuVS Bench when the number has to be compared or reproduced: GPU against GPU, GPU against CPU, a parameter sweep, or the same configuration on another machine. Use `tune` mode when the build/search cross product is too large.

Both sides share the dataset, metric, `k`, batch size, filters, and ground truth. Compare inside one recall bucket, using the Pareto points in that bucket, and benchmark the unit that is searched. Buckets, commands, and result columns are in `references/benchmarking.md`.

## Multi-GPU

Use `cuvs.neighbors.mg` when the index does not fit on one GPU, or when a full copy on every GPU raises query throughput. `replicated` copies the index. `sharded` splits it. Search modes, merge modes, and the IVF variants are in `references/api-index.md`.

## Reference Files

- `references/index-selection.md` — Index comparison, parameter defaults, tuning order, symptom map
- `references/benchmarking.md` — When to run cuVS Bench, recall buckets, Pareto comparison, GPU-against-CPU runs
- `references/ann-patterns.md` — Serving pipeline, integration shapes, filtering, memory estimate
- `references/api-index.md` — Full API index: all index types, multi-GPU module, filtering, quantization, pairwise distance, K-means, IndexParams/SearchParams quick maps
- `references/version-boundaries.md` — Release-specific API and dtype boundaries, especially 24.04 vs 26.06 Python coverage

## External Documentation

Consult the official documentation to retrieve detailed API signatures,
parameter descriptions, and examples on demand.

- **Documentation:** https://docs.nvidia.com/cuvs/home
- **Workload and index choice:** https://docs.nvidia.com/cuvs/getting-started/introduction/vector-search
- **Tuning:** https://docs.nvidia.com/cuvs/getting-started/introduction/tuning-indexes
- **Benchmarking:** https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide
- **Integration Patterns:** https://docs.nvidia.com/cuvs/user-guide/integration-patterns
- **Compatibility:** https://docs.nvidia.com/cuvs/user-guide/compatibility
- **GitHub:** https://github.com/rapidsai/cuvs
- **CHANGELOG:** https://github.com/rapidsai/cuvs/blob/main/CHANGELOG.md
