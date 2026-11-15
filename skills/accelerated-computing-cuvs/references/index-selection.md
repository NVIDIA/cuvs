# cuVS Index Selection Guide

## Index Comparison

| Index | Build Speed | Memory | Recall | Search Speed | Best Use Case |
|---|---|---|---|---|---|
| **CAGRA** | Slower than IVF | High | Very High | Fastest | High-recall GPU search, or GPU build for CPU HNSW serving |
| **IVF-Flat** | Fast | High | Highest (near-exact) | Fast | Smaller datasets, high quality required |
| **IVF-SQ / scalar quantization** | Fast | Lower than IVF-Flat | High, small loss vs flat | Often faster than IVF-Flat | Bandwidth-bound search. Confirm an IVF-SQ index exists before importing it. Otherwise use `cuvs.preprocessing.quantize.scalar` or a lighter IVF-PQ. |
| **IVF-PQ** | Medium | Low (compressed) | Good (tunable) | Fast | Index size is the bottleneck |
| **HNSW** | Medium | Medium | High | Tunable | HNSW serving, GPU hierarchy search, or CPU mutable index handoff |
| **Brute Force** | None | High | Exact | Slowest | Exact top-k, exact filtered/callback parity, and ground truth |

Compare ANN indexes at equivalent recall. For benchmarks and production
selection, build a Pareto view across recall buckets 80–89%, 90–94%,
95–98%, and 99%+, then compare latency, throughput, build time, and memory
within the bucket that matches the product requirement. State `k`, batch size,
and hardware with the numbers. For a reproducible sweep, use cuVS Bench
(`https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide`).

Pareto dimensions for cuVS ANN work:

| Dimension | Measure |
|---|---|
| Search quality | Recall@k, precision@k, NDCG, or task-specific acceptance rate against exact ground truth |
| Serving performance | p50/p95/p99 latency and query throughput at the target batch size |
| Build/refresh cost | Index build time, update time, and refresh cadence |
| Memory footprint | Index size plus raw vectors, temporary build memory, and peak GPU memory |
| Integration cost | GPU-serving path vs GPU-build/CPU-serve handoff, ABI boundary, and operational complexity |

A configuration is Pareto-optimal when no other tested configuration gives at
least the same quality with better latency/throughput, build cost, and memory.

## CAGRA — Graph-Based ANN (High-Throughput Default)

CAGRA builds a proximity graph on GPU (based on the CAGRA algorithm). Best for:
- Static embeddings (build once, search many times)
- Highest throughput requirements
- Datasets of 100K–100M+ vectors

**Build parameters:**
```python
from cuvs.neighbors import cagra

params = cagra.IndexParams(
    metric="sqeuclidean",           # sqeuclidean (default), inner_product
    intermediate_graph_degree=128,  # higher = better quality, slower build, more memory
    graph_degree=64,                # final graph degree — affects index size
    build_algo="ivf_pq",           # "nn_descent" when build time dominates; "ace" when the dataset does not fit on the GPU
    compression=None,               # optional IVF-PQ compression for large datasets
)
index = cagra.build(params, dataset)
```

**Search parameters:**
```python
# Few queries: latency path. algo="auto" also selects from the query count.
search_params = cagra.SearchParams(itopk_size=64, algo="multi_cta")

# Batched throughput. persistent applies only to single_cta.
search_params = cagra.SearchParams(
    itopk_size=64,
    algo="single_cta",
    persistent=True,
)
distances, neighbors = cagra.search(search_params, index, queries, k=10)
```

Use `multi_cta` for a few queries. Use `single_cta` once a batch is larger than about 10. Set `persistent=True` for many concurrent searches, and only with `single_cta`.

**Tuning order (CAGRA):**
1. Raise `itopk_size` until recall is in the target bucket.
2. If recall is still short, raise `graph_degree`.
3. If graph quality is still short, raise `intermediate_graph_degree`. This makes the build more expensive.
4. If build time dominates, set `build_algo="nn_descent"` before growing the graph.
5. If the dataset does not fit in GPU memory, set `build_algo="ace"` or move to IVF-PQ.

No `itopk_size` guarantees a recall. Sweep it on the user's data and report the bucket.

## IVF-Flat — Inverted File Index (High Recall)

Clusters dataset into N lists; searches only nearby lists. Near-exact quality with GPU speed.

```python
from cuvs.neighbors import ivf_flat

params = ivf_flat.IndexParams(
    n_lists=1024,           # aim for about 1,000–10,000 vectors per list
    metric="sqeuclidean",
)
index = ivf_flat.build(params, dataset)

search_params = ivf_flat.SearchParams(
    n_probes=50,            # how many clusters to search — higher = better recall, slower
)
distances, neighbors = ivf_flat.search(search_params, index, queries, k=10)
```

**Tuning order (IVF-Flat):**
1. Set `n_lists` so each list holds about 1,000–10,000 vectors.
2. Raise `n_probes` until recall is in the target bucket.
3. A higher `n_probes / n_lists` ratio improves recall and slows search.

## IVF-PQ — Compressed Large-Scale Search

Product quantization compresses vectors. Use when the full-vector index plus
graph/list overhead does not fit the serving GPU memory budget.

```python
import cupy as cp
from cuvs.neighbors import ivf_pq

params = ivf_pq.IndexParams(
    n_lists=4096,           # aim for about 1,000–10,000 vectors per list
    pq_dim=64,              # compressed dims — must divide into n_features
    pq_bits=8,              # bits per subquantizer (4 or 8)
    metric="sqeuclidean",
)
index = ivf_pq.build(params, dataset)

search_params = ivf_pq.SearchParams(
    n_probes=100,
    internal_distance_dtype=cp.float32,  # float32 for quality, uint8 for speed
    lut_dtype=cp.float32,
)
distances, neighbors = ivf_pq.search(search_params, index, queries, k=10)
```

**Tuning order (IVF-PQ):**
1. Set `n_lists` so each list holds about 1,000–10,000 vectors. More lists make each list smaller and usually need a higher `n_probes` to hold recall.
2. Raise `n_probes` until recall is in the target bucket.
3. Then change compression. Lower `pq_bits` and `pq_dim` reduce memory and can speed search, and they also lose recall.
4. If recall drops, raise `pq_bits` or `pq_dim`, or overfetch and `refine`.
5. Change `lut_dtype`, `internal_distance_dtype`, or `coarse_search_dtype` only after the recall and latency balance is acceptable.

**Memory savings with IVF-PQ:**
- float32 baseline: 128 dims × 4 bytes = 512 bytes/vector
- IVF-PQ (pq_dim=32, pq_bits=8): 32 bytes/vector → 16× compression

**Memory-constrained serving pattern:**
- Use IVF-PQ when projected full-vector storage plus graph/index overhead exceeds the serving GPU budget.
- Keep exact search only for sampled ground truth. The serving path should build/search IVF-PQ, not a full pairwise matrix.
- Recover recall by raising `n_probes`, using higher-quality lookup tables, and overfetching candidates for `refine`.

```python
from cuvs.neighbors import ivf_pq, refine

index = ivf_pq.build(
    ivf_pq.IndexParams(n_lists=4096, pq_dim=64, pq_bits=8, metric="sqeuclidean"),
    dataset,
)
search_params = ivf_pq.SearchParams(n_probes=128)

# Search more candidates than the final k, then exact-rerank just those IDs.
_, candidates = ivf_pq.search(search_params, index, queries, k=40)
distances, neighbors = refine(dataset, queries, candidates, k=10)
```

## Brute Force — Exact Semantics

Use brute force to preserve exact nearest-neighbor behavior, exact filtered
top-k, hnswlib-style callback parity, and recall measurement on a
representative sample. It is a valid GPU production path when the user values
exactness over ANN throughput or the filtered candidate set is modest. For
large serving workloads, use it as the reference path for CAGRA, IVF-Flat, or
IVF-PQ recall validation.

```python
from cuvs.neighbors import brute_force

gt_index = brute_force.build(sample_dataset, metric="sqeuclidean")
gt_distances, gt_neighbors = brute_force.search(gt_index, sample_queries, k=10)
```

## Workload Guidance

| Workload | Start here |
|---|---|
| Under 100K vectors | Brute force or CPU HNSW, unless dimensions or query batches are large |
| Under 1M vectors, exact results, vectors fit | GPU brute force |
| Under 1M vectors, latency matters more than exact recall | CAGRA or HNSW |
| Large dataset, fast ingest | IVF-Flat, scalar quantization, or IVF-PQ |
| Large dataset, high recall, index fits on the GPU | CAGRA |
| Index build is the bottleneck, search stays on CPU | CAGRA build, then `hnsw.from_cagra` |
| Dataset does not fit in memory | DiskANN / Vamana. This is outside an in-memory cuVS index. |

## Where the Time Goes

| Symptom | Change |
|---|---|
| Recall is short and latency is acceptable | CAGRA `itopk_size`, or IVF `n_probes` |
| Recall is still short after search tuning | `graph_degree`, then `intermediate_graph_degree`. For IVF-PQ, raise `pq_dim` / `pq_bits` or add `refine`. |
| Build time dominates | CAGRA `build_algo="nn_descent"` before a larger graph |
| Vectors do not fit on one GPU | `build_algo="ace"`, IVF-PQ, or multi-GPU `mode="sharded"` |
| A few queries look slower than CPU | `algo="multi_cta"`. Stop timing a host round trip per query. |
| Large-batch throughput is low | `algo="single_cta"`, then `persistent=True` |
| Bandwidth-bound, recall is still close | Scalar quantization before IVF-PQ |
| Hash-sharded database | Tune one production segment. Every query searches every shard. |

## Index Persistence

```python
import cuvs

# Save index to disk
cuvs.neighbors.cagra.save("my_index.cagra", index)

# Load index from disk
index = cuvs.neighbors.cagra.load("my_index.cagra")
```
