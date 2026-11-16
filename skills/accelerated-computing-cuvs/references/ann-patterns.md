# cuVS ANN Production Patterns

## Full Pipeline: Build → Search → Use

```python
import cupy as cp
from cuvs.neighbors import cagra

import numpy as np
raw_embeddings = np.load("embeddings.npy").astype(np.float32)  # (N, D)
dataset = cp.asarray(raw_embeddings)

# metric="cosine" when IndexParams accepts it. Fallback is in SKILL.md.
index_params = cagra.IndexParams(
    metric="cosine",
    intermediate_graph_degree=128,
    graph_degree=64,
)
index = cagra.build(index_params, dataset)

import cuvs
cuvs.neighbors.cagra.save("embeddings.cagra", index)

# Later, in serving.
index = cuvs.neighbors.cagra.load("embeddings.cagra")

# One query. Batch search uses algo="single_cta"; see SKILL.md Search Shape.
query = cp.asarray(query_embedding.astype(np.float32).reshape(1, -1))
search_params = cagra.SearchParams(itopk_size=64, algo="multi_cta")
distances, neighbors = cagra.search(search_params, index, query, k=10)

result_ids = cp.asnumpy(neighbors[0])          # shape: (k,)
result_scores = cp.asnumpy(distances[0])       # shape: (k,)
```

## Product Integration Patterns

cuVS can be used as a standalone Python/C++ vector-search library or embedded
behind a database, search engine, or application service. Choose the integration
shape before writing code:

| Pattern | Use When | Typical Shape |
|---|---|---|
| Direct library integration | The product owns GPU resources, batching, and index lifecycle | Link cuVS in-process and build/search indexes directly |
| Offloaded GPU index build | Serving remains in an existing database/search stack, but index refresh is the bottleneck | Send build jobs to GPU workers and import the finished artifact |
| GPU-build / CPU-search handoff | CPU search infrastructure is fixed, but HNSW build time blocks ingestion | Build CAGRA on GPU, convert/export to HNSW, serve in the CPU stack |
| ABI-stable C API | The integration needs binary compatibility across packaged plugins or language bindings | Put cuVS behind the C API and track the ABI compatibility window |

For a hash-sharded database, tune one segment and time the fan-out separately. See `references/benchmarking.md`.

Use this shape for the recommendation:

| Section | Include |
|---|---|
| Pattern | Name the chosen cuVS integration shape and the fallback/next option. |
| Why | Tie the choice to refresh SLA, query latency, GPU placement, memory, and artifact constraints. |
| Compatibility Gate | Installed cuVS version, C API ABI window, CUDA/driver/toolkit, deployment hardware, plugin/package boundary, and artifact format. |
| Validation Gate | Recall target, exact-search ground truth, equivalent-recall comparison, build/import/load time, serving p95/p99, memory, and rollback. |
| Boundary | Keep RAFT/custom CUDA for lower-level reusable primitives; keep CPU paths as baseline/fallback when they remain part of production. |

## RAG Pipeline Integration

```python
from cuvs.neighbors import cagra
import cupy as cp

class GPUVectorStore:
    def __init__(self, embeddings: np.ndarray, documents: list, k: int = 5):
        self.documents = documents
        self.k = k

        dataset = cp.asarray(embeddings.astype(np.float32))
        index_params = cagra.IndexParams(
            metric="cosine",
            graph_degree=64,
        )
        self.index = cagra.build(index_params, dataset)
        self.search_params = cagra.SearchParams(itopk_size=64, algo="multi_cta")

    def retrieve(self, query_embedding: np.ndarray) -> list[str]:
        q = cp.asarray(query_embedding.astype(np.float32).reshape(1, -1))

        _, neighbors = cagra.search(self.search_params, self.index, q, k=self.k)
        ids = cp.asnumpy(neighbors[0])
        return [self.documents[i] for i in ids]
```

## Filtering Search Results

A bitset filter is a packed `uint32` array where bit `1` means that dataset row is allowed. Rule 9 in `SKILL.md` chooses `prefilter=` or `filter=`.

```python
import numpy as np
import cupy as cp
from cuvs.neighbors import brute_force, cagra, filters

def make_bitset(valid: np.ndarray) -> cp.ndarray:
    words = np.zeros((valid.size + 31) // 32, dtype=np.uint32)
    for idx in np.flatnonzero(valid):
        words[int(idx) // 32] |= np.uint32(1) << np.uint32(int(idx) % 32)
    return cp.asarray(words)

valid = np.ones(len(dataset), dtype=bool)
valid[excluded_ids] = False
prefilter = filters.from_bitset(make_bitset(valid))

# Exact filtered top-k.
exact_index = brute_force.build(dataset, metric="sqeuclidean")
distances, neighbors = brute_force.search(
    exact_index, queries, k=10, prefilter=prefilter
)

# Approximate filtered serving path.
search_params = cagra.SearchParams(itopk_size=64)
ann_distances, ann_neighbors = cagra.search(
    search_params, index, queries, k=10, filter=prefilter
)
```

## Memory Estimation

```python
# CAGRA index memory (approximate)
n_vectors = 1_000_000
n_dims = 128
graph_degree = 64

# Float32 vectors
vector_memory = n_vectors * n_dims * 4   # bytes
# Graph adjacency
graph_memory = n_vectors * graph_degree * 4

total_gb = (vector_memory + graph_memory) / 1e9
print(f"Estimated index memory: {total_gb:.1f} GB")
```
