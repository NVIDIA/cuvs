---
slug: api-reference/python-api-neighbors
---

# Neighbors

_Python module: `cuvs.neighbors`_

## refine

`@auto_sync_resources`
`@auto_convert_output`

```python
def refine(dataset, queries, candidates, k=None, metric="sqeuclidean", indices=None, distances=None, resources=None)
```

Refine nearest neighbor search.

Refinement is an operation that follows an approximate NN search. The
approximate search has already selected n_candidates neighbor candidates
for each query. We narrow it down to k neighbors. For each query, we
calculate the exact distance between the query and its n_candidates
neighbor candidate, and select the k nearest ones.

Input arrays can be either CUDA array interface compliant matrices or
array interface compliant matrices in host memory. All array must be in
the same memory space.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dataset` | `array interface compliant matrix, shape (n_samples, dim)` | Supported dtype [float32, int8, uint8, float16] |
| `queries` | `array interface compliant matrix, shape (n_queries, dim)` | Supported dtype [float32, int8, uint8, float16] |
| `candidates` | `array interface compliant matrix, shape (n_queries, k0)` | Supported dtype int64 |
| `k` | `int` | Number of neighbors to search (k &lt;= k0). Optional if indices or distances arrays are given (in which case their second dimension is k). |
| `metric` | `str` | Name of distance metric to use, default ="sqeuclidean" |
| `indices` | `Optional array interface compliant matrix shape            (n_queries, k).` | If supplied, neighbor indices will be written here in-place. (default None). Supported dtype int64. |
| `distances` | `Optional array interface compliant matrix shape              (n_queries, k).` | If supplied, neighbor indices will be written here in-place. (default None) Supported dtype float. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.common import Resources
>>> from cuvs.neighbors import ivf_pq, refine
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> resources = Resources()
>>> index_params = ivf_pq.IndexParams(n_lists=1024,
...                                   metric="sqeuclidean",
...                                   pq_dim=10)
>>> index = ivf_pq.build(index_params, dataset, resources=resources)
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> k = 40
>>> _, candidates = ivf_pq.search(ivf_pq.SearchParams(), index,
...                               queries, k, resources=resources)
>>> k = 10
>>> distances, neighbors = refine(dataset, queries, candidates, k)
```
