# cuVS Python API Quick Reference

## Nearest Neighbor Index Types

| Index | Module |
|---|---|
| **CAGRA** | `cuvs.neighbors.cagra` |
| **IVF-Flat** | `cuvs.neighbors.ivf_flat` |
| **IVF-PQ** | `cuvs.neighbors.ivf_pq` |
| **Brute-Force** | `cuvs.neighbors.brute_force` |
| **HNSW** | `cuvs.neighbors.hnsw` |
| **NN-Descent** | `cuvs.neighbors.nn_descent` |
| **All-Neighbors** | `cuvs.neighbors.all_neighbors` |

### Multi-GPU Indexes (`cuvs.neighbors.mg`)

| Index | Module |
|---|---|
| Multi-GPU CAGRA | `cuvs.neighbors.mg.cagra` |
| Multi-GPU IVF-Flat | `cuvs.neighbors.mg.ivf_flat` |
| Multi-GPU IVF-PQ | `cuvs.neighbors.mg.ivf_pq` |

See [Multi-GPU section](#multi-gpu-vector-search) below.

---

## Core Operations (all indexes follow this pattern)

```python
from cuvs.neighbors import cagra   # substitute ivf_flat, ivf_pq, etc.

# Build
index_params = cagra.IndexParams(...)
index = cagra.build(index_params, dataset)     # dataset: supported 2D array

# Search
search_params = cagra.SearchParams(...)
distances, neighbors = cagra.search(search_params, index, queries, k=10)

# Persist
cagra.save("index.bin", index)
index = cagra.load("index.bin")

# Extend (add new vectors — supported by CAGRA, IVF-Flat, IVF-PQ, HNSW)
index = cagra.extend(index_params, new_vectors, index)
```

---

Tuning order is in `references/index-selection.md`. This map is the parameter names.

## IndexParams Quick Map

| Index | Key Params |
|---|---|
| **CAGRA** | `graph_degree` (64), `intermediate_graph_degree` (128), `metric`, `build_algo` (`nn_descent`, `ace`) |
| **IVF-Flat** | `n_lists` (~1k–10k vectors/list), `metric` |
| **IVF-PQ** | `n_lists`, `pq_dim`, `pq_bits` (8) |
| **HNSW** | `hierarchy` (`"gpu"` default), ACE params, or `from_cagra()` |
| **NN-Descent** | `n_clusters`, `max_iterations` |
| **Brute-Force** | `metric` |

## SearchParams Quick Map

| Index | Key Params |
|---|---|
| **CAGRA** | `itopk_size` (64), `algo` (`multi_cta`, `single_cta`), `persistent` (only with `single_cta`) |
| **IVF-Flat** | `n_probes` |
| **IVF-PQ** | `n_probes`, `lut_dtype`, `internal_distance_dtype`, `coarse_search_dtype` |
| **HNSW** | `ef` |
| **Brute-Force** | none |

---

Dtype tables are in `references/version-boundaries.md`.

## HNSW: GPU Hierarchy or CPU Serve

Current cuVS HNSW uses `hnsw.IndexParams(hierarchy="gpu")` by default. That
keeps most hierarchy work on the GPU for GPU search. Use `hierarchy="cpu"` when
the final serving environment is CPU-only or needs mutable CPU HNSW behavior.

```python
from cuvs.neighbors import cagra, hnsw

# Build with GPU
index = cagra.build(cagra.IndexParams(), dataset)

# Convert CAGRA to HNSW. hierarchy="gpu" is the default.
hnsw_index = hnsw.from_cagra(hnsw.IndexParams(hierarchy="gpu"), index)

# Search
search_params = hnsw.SearchParams(ef=200)
distances, neighbors = hnsw.search(search_params, hnsw_index, queries, k=10)

# Save/load. Use hierarchy="cpu" if the loaded index must be mutable on CPU.
hnsw.save("index.hnsw", hnsw_index)
hnsw_index = hnsw.load("index.hnsw", hnsw.SearchParams(), n_dims=128, dtype=np.float32)
```

For datasets too large to fit on one GPU during build, use HNSW ACE parameters
instead of assuming the full dataset must be resident at once.

---

## Multi-GPU Vector Search

`cuvs.neighbors.mg` provides multi-GPU variants of CAGRA, IVF-Flat, and IVF-PQ. Each index is distributed across all available GPUs.

```python
from cuvs.neighbors.mg import cagra as mg_cagra
import cupy as cp

# Build across all GPUs (data automatically distributed)
index_params = mg_cagra.IndexParams(
    mode="replicated",    # "replicated" (all GPUs search full index)
                          # "sharded" (each GPU holds a partition)
)
index = mg_cagra.build(index_params, dataset)

# Search (queries split across GPUs automatically)
search_params = mg_cagra.SearchParams(
    itopk_size=64,
    search_mode="load_balancer",   # or "round_robin"
    merge_mode="merge_on_root_rank",  # or "tree_merge"
)
distances, neighbors = mg_cagra.search(search_params, index, queries, k=10)

# Serialize
mg_cagra.save("mg_index/", index)
index = mg_cagra.load("mg_index/")

# Distribute pre-built single-GPU index across GPUs
mg_index = mg_cagra.distribute(index)
```

### Multi-GPU Distribution Modes

| Mode | When to Use |
|---|---|
| `replicated` | Each GPU holds full index; good for read-heavy query loads |
| `sharded` | Index partitioned across GPUs; scales to datasets larger than one GPU's VRAM |

### Multi-GPU Search Modes

| Mode | Behavior |
|---|---|
| `load_balancer` | Distributes query batch adaptively across GPUs |
| `round_robin` | Rotates queries across GPUs in fixed order |

### Multi-GPU Merge Modes

| Mode | Behavior |
|---|---|
| `merge_on_root_rank` | All partial results sent to rank 0 for merge |
| `tree_merge` | Reduces along a binary tree — better for many GPUs |

---

## Additional Modules

### Filtering

`filters.from_bitset` and `filters.from_bitmap` build the pre-search filter. Brute force takes it as `prefilter=`. CAGRA, IVF-Flat, and IVF-PQ take it as `filter=`. The bitset layout and a full example are in `references/ann-patterns.md`.

### Refinement

`refine(dataset, queries, candidates, k)` exact-reranks an overfetched candidate list. The IVF-PQ example is in `references/index-selection.md`.

### Pairwise Distance

Compute all pairwise distances between two sets of vectors:

```python
from cuvs.distance import pairwise_distance

# a: (m, d), b: (n, d) → output: (m, n)
distances = pairwise_distance(a, b, metric="euclidean")
# Metrics: euclidean, sqeuclidean, cosine, inner_product, l1, l2, ...
```

### K-Means Clustering

```python
from cuvs.cluster import kmeans

params = kmeans.KMeansParams(n_clusters=100, max_iter=300)
centroids, inertia, n_iter = kmeans.fit(params, dataset)
labels = kmeans.predict(params, centroids, dataset)
```

### Preprocessing / Quantization

Pre-quantize vectors to reduce memory or speed up index builds:

```python
# Product Quantizer (standalone, independent of IVF-PQ index)
from cuvs.preprocessing.quantize import pq

params = pq.QuantizerParams(pq_dim=32, pq_bits=8)
quantizer = pq.build(params, training_data)     # train on a sample
compressed = pq.transform(quantizer, dataset)   # compress all vectors
reconstructed = pq.inverse_transform(quantizer, compressed)

# Scalar Quantizer
from cuvs.preprocessing.quantize import scalar

params = scalar.QuantizerParams()
quantizer = scalar.train(params, training_data)
compressed = scalar.transform(quantizer, dataset)

# Binary Quantization (fastest, lowest quality)
from cuvs.preprocessing.quantize import binary
binary_vectors = binary.transform(dataset)   # each float → 1 bit
```

---

## Module Import Summary

```python
# Single-GPU indexes
from cuvs.neighbors import cagra, ivf_flat, ivf_pq, brute_force, hnsw
from cuvs.neighbors import nn_descent, all_neighbors

# Multi-GPU indexes
from cuvs.neighbors.mg import cagra as mg_cagra
from cuvs.neighbors.mg import ivf_flat as mg_ivf_flat
from cuvs.neighbors.mg import ivf_pq as mg_ivf_pq

# Utilities
from cuvs.neighbors import filters   # filters.from_bitset/from_bitmap/no_filter
from cuvs.neighbors import refine
from cuvs.distance import pairwise_distance
from cuvs.cluster import kmeans
from cuvs.preprocessing.quantize import pq, scalar, binary
```
