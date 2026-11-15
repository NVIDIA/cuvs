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

- Release tracked by this skill: 26.06.
- Requires an NVIDIA GPU and a matching RAPIDS/cuVS install for the target CUDA/Python environment. This skill targets the planned 26.06 cuVS release line; verify final platform support with the RAPIDS install selector when the release is published. Dtype support is API- and release-specific; use `references/version-boundaries.md` when the installed version or docs differ from 26.06.

## Naming

Use NVIDIA library-first wording in user-facing answers. Keep literal RAPIDS/rapidsai URLs, package names, and release metadata when citing sources.

## Role

You are a cuVS expert helping an implementer build GPU-accelerated vector search pipelines, or helping a database/search-platform integrator choose how cuVS fits behind their service. The user understands their embedding data and search requirements — your job is to select the right index or integration pattern, configure it correctly, and wire it into their pipeline. Vector search quality vs speed is a tradeoff that requires deliberate configuration.

## Critical Rules

1. **Choose the index for the workload.** CAGRA for high-recall search that fits on the GPU. IVF-Flat when full-precision lists are enough. Scalar quantization when bandwidth is the limit and the recall loss can stay small. IVF-PQ when index size is the limit. See `references/index-selection.md`.
2. **Handle dtype deliberately.** `float32` is the safe baseline for dense embeddings and exact ground-truth paths. Several 26.06 Python index APIs also accept `float16`, `int8`, or `uint8`; preserve those dtypes when the chosen API supports them and the data is already quantized or memory-bound. Validate recall after changing dtype.
3. **Three dimensions to balance:** build time, search quality (recall), search throughput. You cannot maximize all three simultaneously.
4. **Tune search before build.** CAGRA: `itopk_size`, then `graph_degree`, then `intermediate_graph_degree`. IVF: about 1,000–10,000 vectors per list, then `n_probes`, then compression, then `refine`. Compare latency, throughput, build time, and memory inside one recall bucket: 80–89%, 90–94%, 95–98%, or 99%+. For a hash-sharded database, tune one production segment, not the full corpus. See `references/index-selection.md`.
5. **Match exactness before speed.** Use cuVS brute force for exact top-k,
   exact filtered search, callback-filter parity, and recall ground truth. Use
   ANN indexes when approximate recall is acceptable and measured.
6. **Place the GPU where the workload benefits.** Under about 100K vectors, start with brute force or CPU HNSW unless dimensions or query batches are large. Use IVF-Flat, scalar quantization, or IVF-PQ when a large dataset needs fast ingest. Use CAGRA when a high-recall index fits in GPU memory. Use a GPU build with CPU HNSW serving when indexing is the bottleneck and search stays on CPU. A corpus that does not fit in memory is a DiskANN problem, not an in-memory cuVS index. When the production path is GPU search, report CPU timing separately.
7. **Use memory-aware ANN when the full-vector path does not fit.** Prefer scalar quantization when the limit is bandwidth, then IVF-PQ or compressed CAGRA when the limit is index size. Recover recall with larger `n_probes`/`itopk_size` and overfetch plus `refine`. Confirm an IVF-SQ index exists in the installed package before importing one; otherwise use `cuvs.preprocessing.quantize.scalar`.
8. **Keep cuVS inputs device-resident.** Prefer CuPy, PyTorch CUDA tensors, or other `__cuda_array_interface__` arrays. NumPy host arrays require a visible host→device transfer step before build/search.
9. **Apply filters before search.** For filtered vector search, use cuVS
   pre-search filters (`filters.from_bitset` or `filters.from_bitmap`) so the
   search ranks only eligible vectors. Exact filtered top-k uses
   `brute_force.search(..., prefilter=...)`; ANN filtered search uses
   `search(..., filter=...)` and needs recall validation against the exact path.
10. **Stay at the vector-search layer.** For embedding search services, use cuVS indexes. Use RAFT for lower-level primitives and custom library work when the task is below the index/service layer.
11. **Use the C API for binary-stable product integrations.** For vector databases, search engines, language bindings, or packaged apps that need ABI stability, choose the cuVS C API boundary and check the installed ABI window. See `references/ann-patterns.md` and `references/version-boundaries.md`.

## Index Selection Quick Guide

| Index | Best For | Recall | Build Speed | Search Speed |
|---|---|---|---|---|
| **CAGRA** | High-recall search that fits on the GPU, or a GPU build for CPU HNSW serving | Very High | Slower than IVF | Fastest |
| **IVF-Flat** | Full-precision lists, fast ingest | Highest | Fastest | Fast |
| **Scalar quantization** | Bandwidth-bound search with a small recall loss | High | Fast | Fast |
| **IVF-PQ** | Index size is the limit | High (tunable) | Fast | Fast |
| **HNSW** | HNSW search with GPU hierarchy by default, or CPU serving after GPU build | High | Tunable | GPU or CPU |
| **Brute Force** | Exact top-k, exact filtered/callback parity, validation | Exact | Fastest | Slower |

Start from the workload. Under about 100K vectors, start with brute force or CPU HNSW. For a large corpus that must ingest quickly, start with IVF-Flat or IVF-PQ. For high-recall search that fits on the GPU, start with CAGRA. Start with brute force when the user needs exact nearest-neighbor semantics, exact metadata-filtered top-k, or hnswlib-style callback parity. See `references/index-selection.md`.

## Search Shape

Pick the CAGRA search kernel from the query count before tuning `itopk_size`.

```python
from cuvs.neighbors import cagra

# Few queries: latency path. algo="auto" also selects from the query count.
search_params = cagra.SearchParams(itopk_size=64, algo="multi_cta")

# Batched throughput. persistent applies only to single_cta.
search_params = cagra.SearchParams(
    itopk_size=64,
    algo="single_cta",
    persistent=True,
)
```

Use `multi_cta` for a few queries. Use `single_cta` once a batch is larger than about 10. Set `persistent=True` for a service with many concurrent searches, and only with `single_cta`. A Python loop that searches one vector and copies each result to the host measures the round trip, not GPU search.

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

## IVF-PQ (Compressed Large-Scale Search)

```python
from cuvs.neighbors import ivf_pq, refine

# Build
index_params = ivf_pq.IndexParams(
    n_lists=1024,      # aim for about 1,000–10,000 vectors per list
    pq_dim=32,         # compressed dims — lower = smaller index, lower recall
    pq_bits=8,
)
index = ivf_pq.build(index_params, dataset)

# Search
search_params = ivf_pq.SearchParams(n_probes=50)  # higher = better recall, slower
distances, neighbors = ivf_pq.search(search_params, index, queries, k=10)

# Optional recall recovery: overfetch approximate candidates, then exact-rerank
# only those candidates against the original dataset.
_, candidates = ivf_pq.search(search_params, index, queries, k=40)
distances, neighbors = refine(dataset, queries, candidates, k=10)
```

## IVF-Flat (High Recall, Smaller Datasets)

```python
from cuvs.neighbors import ivf_flat

index_params = ivf_flat.IndexParams(n_lists=1024)
index = ivf_flat.build(index_params, dataset)

search_params = ivf_flat.SearchParams(n_probes=50)
distances, neighbors = ivf_flat.search(search_params, index, queries, k=10)
```

## Interoperability

cuVS is happiest with device arrays. Use CuPy or framework GPU tensors for
build/search inputs and convert outputs deliberately for reporting:

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

## Dtype Guidance

Use `float32` for a first implementation unless the workload already has a
lower-precision embedding representation or the memory budget requires one.
For 26.06 Python APIs, common index families support these input dtypes:

| API | 26.06 Python input dtype guidance |
|---|---|
| CAGRA | `float32`, `float16`, `int8`, `uint8` |
| IVF-Flat | `float32`, `float16`, `int8`, `uint8` |
| IVF-PQ | `float32`, `float16`, `int8`, `uint8`; tune `lut_dtype`, `internal_distance_dtype`, and `coarse_search_dtype` deliberately |
| HNSW | `float32`, `float16`, `int8`, `uint8` |
| Brute force | `float32` or `float16` |
| Refine | `float32`, `float16`, `int8`, `uint8` inputs with `float32` distances |

When a task depends on a specific release, read `references/version-boundaries.md`
before changing dtype assumptions or API calls.

## Cosine Similarity

CAGRA and IVF-Flat accept `metric="cosine"` in the current public Python API. Use it when the installed `IndexParams` documents `cosine`.

```python
index_params = cagra.IndexParams(metric="cosine")
index = cagra.build(index_params, dataset)
```

When the installed version does not accept `cosine`, L2-normalize the dataset and the queries, then use `metric="inner_product"`. Check `references/version-boundaries.md` before choosing.

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

Compare p50/p95/p99 latency, throughput, build time, and memory only inside one recall bucket: 80–89%, 90–94%, 95–98%, or 99%+. State `k`, batch size, and hardware with the numbers. A migration is ready when that bucket meets the recall target and the serving latency budget.

For a reproducible sweep, use [cuVS Bench](https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/cu-vs-bench-tool/introduction). It records build time, throughput, latency, and recall, and its tune mode searches parameters against a target. Follow the [benchmark methodology](https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/methodologies).

## Multi-GPU Vector Search

For datasets that exceed one GPU's memory, or for higher query throughput, cuVS provides multi-GPU variants of all three major index types via `cuvs.neighbors.mg`:

```python
from cuvs.neighbors.mg import cagra as mg_cagra

# Build across all visible GPUs
index_params = mg_cagra.IndexParams(mode="replicated")  # or "sharded"
index = mg_cagra.build(index_params, dataset)

search_params = mg_cagra.SearchParams(itopk_size=64)
distances, neighbors = mg_cagra.search(search_params, index, queries, k=10)
```

- `mode="replicated"` — every GPU holds the full index; best for high query throughput
- `mode="sharded"` — index is split across GPUs; scales to datasets larger than one GPU's VRAM

See `references/api-index.md` for multi-GPU search modes, merge modes, and IVF-Flat/IVF-PQ multi-GPU variants.

## Troubleshooting

**Low recall:**
- Raise the search knob first: `itopk_size` for CAGRA, `n_probes` for IVF-Flat and IVF-PQ.
- If recall is still short, raise `graph_degree`, then `intermediate_graph_degree`.
- For IVF-PQ, raise `pq_dim` or `pq_bits`, or overfetch and `refine`.
- See `references/index-selection.md` for the full order.

**Latency is high for a few queries:**
- Use `algo="multi_cta"`, or leave `algo="auto"`.
- A one-query Python loop that copies each result to the host measures the round trip, not GPU search.

**Throughput is low on large batches:**
- Set `algo="single_cta"`. For many concurrent searches, also set `persistent=True`.
- Then raise `itopk_size` if recall is short.

**Build time dominates:**
- Switch CAGRA `build_algo` to `"nn_descent"` before growing the graph.

**OOM during index build:**
- If the graph is the problem, reduce `intermediate_graph_degree`.
- If the vectors do not fit on the GPU, use `build_algo="ace"` or IVF-PQ.
- A dataset that does not fit in host memory is a DiskANN problem.
- If allocation fragmentation is part of the failure, configure an RMM async memory resource before cuVS allocations.

**A hash-sharded database is slow end to end:**
- Tune and measure one production segment. Every query searches every shard, so a faster single index can disappear after the fan-out and merge.

**Wrong results with cosine similarity:**
- Use `metric="cosine"` when the installed API accepts it.
- Otherwise normalize both the dataset and the queries before `inner_product`.

## Reference Files

- `references/index-selection.md` — Detailed index comparison, when to use each, tuning guide
- `references/ann-patterns.md` — Production patterns, batching, index persistence, integration examples
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
