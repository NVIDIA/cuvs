# cuVS ANN Production Patterns

## Full Pipeline: Build → Search → Use

```python
import cupy as cp
from cuvs.neighbors import cagra

# 1. Prepare embeddings. float32 is the safe dense default; preserve a supported
# lower-precision dtype only when the chosen cuVS API and recall target support it.
import numpy as np
raw_embeddings = np.load("embeddings.npy").astype(np.float32)  # (N, D)

# 2. Move to GPU
dataset = cp.asarray(raw_embeddings)

# 3. Build index. Use metric="cosine" when IndexParams accepts it.
#    Fallback: L2-normalize the dataset and the queries, then metric="inner_product".
index_params = cagra.IndexParams(
    metric="cosine",
    intermediate_graph_degree=128,
    graph_degree=64,
)
index = cagra.build(index_params, dataset)

# 4. Save index for reuse
import cuvs
cuvs.neighbors.cagra.save("embeddings.cagra", index)

# --- Later, in serving ---

# 5. Load index
index = cuvs.neighbors.cagra.load("embeddings.cagra")

# 6. Search one query: small-batch kernel. Batch queries use algo="single_cta".
query = cp.asarray(query_embedding.astype(np.float32).reshape(1, -1))
search_params = cagra.SearchParams(itopk_size=64, algo="multi_cta")
distances, neighbors = cagra.search(search_params, index, query, k=10)

# 7. Get results as numpy for downstream use
result_ids = cp.asnumpy(neighbors[0])          # shape: (k,)
result_scores = cp.asnumpy(distances[0])       # shape: (k,)
```

## Batched Search for Throughput

```python
# For large query batches, send all queries at once (not one-by-one)
all_queries = cp.asarray(queries_numpy.astype(np.float32))  # (n_queries, n_dims)

search_params = cagra.SearchParams(itopk_size=64, algo="single_cta")
distances, neighbors = cagra.search(search_params, index, all_queries, k=10)

# Copy the whole result once.
ids = cp.asnumpy(neighbors)
scores = cp.asnumpy(distances)
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

For vector database or search-platform work, decide whether the user wants a
query-time GPU serving path, a faster index-build path, or a stable product ABI.
That choice should drive API selection as much as CAGRA/IVF parameter tuning.

For a hash-sharded or segment-based database, tune list counts and search
parameters for one production segment. A 100M corpus stored as ten 10M segments
is a 10M-vector problem. End-to-end latency includes searching every segment
and merging the partial top-k, so time a single index separately from the fan-out.

Service-integration recommendations should cover these checks:

- Identify the workload as a cuVS database/search-platform integration when the
  user describes vector indexes, retrieval services, ANN refresh, or embedding
  search behind a product boundary.
- Pick the integration shape from the service constraints: direct GPU serving,
  offloaded GPU index build, GPU-build/CPU-search handoff, or C API boundary.
- For native plugins, language bindings, or packaged database extensions, use
  the cuVS C API as the binary boundary and check the installed cuVS version,
  C API ABI window, CUDA/driver/toolkit compatibility, and deployment hardware.
- Compare candidates at equivalent recall, including index build, serialization,
  artifact import/load, serving p95/p99, memory footprint, and rollback path.
- Keep RAFT or custom CUDA at the reusable-primitive layer unless the user is
  explicitly building lower-level vector-search infrastructure.

Use this compact shape for final recommendations:

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

        # Build index at init. Fallback for builds without metric="cosine":
        # L2-normalize embeddings and queries, then use metric="inner_product".
        dataset = cp.asarray(embeddings.astype(np.float32))
        index_params = cagra.IndexParams(
            metric="cosine",
            graph_degree=64,
        )
        self.index = cagra.build(index_params, dataset)
        # One query per call. Batch queries and use algo="single_cta" for throughput.
        self.search_params = cagra.SearchParams(itopk_size=64, algo="multi_cta")

    def retrieve(self, query_embedding: np.ndarray) -> list[str]:
        q = cp.asarray(query_embedding.astype(np.float32).reshape(1, -1))

        _, neighbors = cagra.search(self.search_params, self.index, q, k=self.k)
        ids = cp.asnumpy(neighbors[0])
        return [self.documents[i] for i in ids]
```

## Recall Validation Pattern

Always validate recall before production deployment. Use exact search only for
ground truth on a representative sample; keep the production path on the
selected ANN index.

```python
from cuvs.neighbors import cagra, brute_force
import cupy as cp

def validate_recall(dataset, queries, index, search_params, k=10) -> float:
    """Recall@k: matched neighbors / (n_queries * k), per query, not one flattened set."""
    gt_index = brute_force.build(dataset, metric="sqeuclidean")
    _, gt_neighbors = brute_force.search(gt_index, queries, k=k)
    _, approx_neighbors = cagra.search(search_params, index, queries, k=k)

    approx = cp.asarray(approx_neighbors)
    truth = cp.asarray(gt_neighbors)
    matches = (approx[:, :, None] == truth[:, None, :]).any(axis=2).sum(axis=1)
    return (matches.sum() / (approx.shape[0] * k)).item()

# Use
dataset = cp.asarray(embeddings.astype("float32"))
index = cagra.build(cagra.IndexParams(), dataset)

sample_queries = dataset[:1000]   # use subset for validation
search_params = cagra.SearchParams(itopk_size=64)

recall = validate_recall(dataset, sample_queries, index, search_params, k=10)
print(f"Recall@10: {recall:.3f}")   # should be > 0.95 for most use cases
```

## Where the GPU Belongs

Choose the path from the workload table in `references/index-selection.md`. Under about 100K vectors, keep brute force or CPU HNSW unless dimensions or query batches are large. Use GPU search when the corpus is large enough and the index fits in GPU memory. When index build is the bottleneck and serving stays on CPU, build CAGRA and export HNSW with `hnsw.from_cagra`, and report that path separately from GPU search and from any CPU baseline.

```python
def search_embeddings(corpus, queries, k, *, backend="gpu"):
    if backend == "cpu":
        return exact_cpu_baseline(corpus, queries, k)
    if backend != "gpu":
        raise ValueError(f"unsupported backend: {backend}")
    return cuvs_index_search(corpus, queries, k)
```

## Approximate Search With Exact Candidate Refinement

For memory-constrained indexes such as IVF-PQ, overfetch candidates and rerank
only those candidates with exact distances. This usually costs far less memory
than a full-vector index or full pairwise distance matrix.

```python
from cuvs.neighbors import ivf_pq, refine

index = ivf_pq.build(
    ivf_pq.IndexParams(n_lists=4096, pq_dim=64, pq_bits=8, metric="sqeuclidean"),
    dataset,
)
search_params = ivf_pq.SearchParams(n_probes=128)

# Ask the ANN index for more than the final k.
_, candidates = ivf_pq.search(search_params, index, queries, k=40)

# Rerank those candidate IDs exactly against the original dataset.
distances, neighbors = refine(dataset, queries, candidates, k=10)
```

Tune `n_probes`, overfetch depth, and `pq_dim` together. Increasing `n_probes`
and overfetch improves recall at search-time cost; increasing `pq_dim` improves
quality at index-memory cost.

## Filtering Search Results

cuVS supports pre-search filtering, which restricts candidates before nearest
neighbor ranking. In Python 26.06, create filters with `filters.from_bitset` or
`filters.from_bitmap`.

Use `brute_force.search(..., prefilter=...)` when the user needs exact filtered
top-k behavior, hnswlib-style callback parity, or a validation baseline. Use
ANN index filtering (`search(..., filter=...)`) when approximate recall is an
acceptable serving tradeoff and compare it against the exact filtered path.

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

`brute_force.search` uses the same filter object as `prefilter=prefilter`; ANN
indexes such as CAGRA, IVF-Flat, and IVF-PQ use `filter=prefilter`.

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
