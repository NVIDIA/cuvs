---
slug: api-reference/python-api-neighbors-ivf-sq
---

# IVF SQ

_Python module: `cuvs.neighbors.ivf_sq`_

## Index

```python
cdef class Index
```

IvfSq index object. This object stores the trained IvfSq index state
which can be used to perform nearest neighbors searches.

**Members**

| Name | Kind |
| --- | --- |
| `trained` | property |
| `n_lists` | property |
| `dim` | property |
| `centers` | property |

### trained

```python
def trained(self)
```

### n_lists

```python
def n_lists(self)
```

The number of inverted lists (clusters)

### dim

```python
def dim(self)
```

dimensionality of the cluster centers

### centers

```python
def centers(self)
```

Get the cluster centers corresponding to the lists in the
original space

## IndexParams

```python
cdef class IndexParams
```

Parameters to build index for IvfSq nearest neighbor search

Note: IVF-SQ currently uses fixed 8-bit residual scalar quantization.
There are no additional SQ-specific tuning knobs.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `n_lists` | `int, default = 1024` | The number of clusters used in the coarse quantizer. |
| `metric` | `str, default = "sqeuclidean"` | String denoting the metric type.<br />Valid values for metric: ["sqeuclidean", "inner_product", "euclidean", "cosine"], where<br /><br />- sqeuclidean is the euclidean distance without the square root operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,<br />- euclidean is the euclidean distance<br />- inner product distance is defined as distance(a, b) = \\sum_i a_i * b_i.<br />- cosine distance is defined as distance(a, b) = 1 - \\sum_i a_i * b_i / ( \|\|a\|\|_2 * \|\|b\|\|_2). |
| `metric_arg` | `float, default = 2.0` | Additional metric argument forwarded to cuVS distance computations. |
| `kmeans_n_iters` | `int, default = 20` | The number of iterations searching for kmeans centers during index building. |
| `max_train_points_per_cluster` | `int, default = 256` | The number of data vectors per cluster to use during iterative kmeans building. The index uses at most n_lists * max_train_points_per_cluster rows for training. |
| `add_data_on_build` | `bool, default = True` | After training the coarse clustering model and residual scalar quantization parameters, we populate the index with the dataset if add_data_on_build == True. Otherwise, the index is left empty, and the extend method can be used to add new vectors to the index. |
| `conservative_memory_allocation` | `bool, default = False` | By default, the algorithm allocates more space than necessary for individual clusters (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of data copies during repeated calls to `extend` (extending the database). To disable this behavior and use as little GPU memory for the database as possible, set this flag to `True`. |

**Constructor**

```python
def __init__(self, *, n_lists=1024, metric="sqeuclidean", metric_arg=2.0, kmeans_n_iters=20, max_train_points_per_cluster=256, add_data_on_build=True, conservative_memory_allocation=False)
```

**Members**

| Name | Kind |
| --- | --- |
| `get_handle` | method |
| `metric` | property |
| `metric_arg` | property |
| `add_data_on_build` | property |
| `n_lists` | property |
| `kmeans_n_iters` | property |
| `max_train_points_per_cluster` | property |
| `conservative_memory_allocation` | property |

### get_handle

```python
def get_handle(self)
```

### metric

```python
def metric(self)
```

### metric_arg

```python
def metric_arg(self)
```

### add_data_on_build

```python
def add_data_on_build(self)
```

### n_lists

```python
def n_lists(self)
```

### kmeans_n_iters

```python
def kmeans_n_iters(self)
```

### max_train_points_per_cluster

```python
def max_train_points_per_cluster(self)
```

### conservative_memory_allocation

```python
def conservative_memory_allocation(self)
```

## SearchParams

```python
cdef class SearchParams
```

Supplemental parameters to search IVF-SQ index

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `int` | The number of clusters to search. |

**Constructor**

```python
def __init__(self, *, n_probes=20)
```

**Members**

| Name | Kind |
| --- | --- |
| `get_handle` | method |
| `n_probes` | property |

### get_handle

```python
def get_handle(self)
```

### n_probes

```python
def n_probes(self)
```

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, resources=None)
```

Build the IvfSq index from the dataset for efficient search.

IVF-SQ (Scalar Quantization) uses IVF partitioning together with
per-dimension scalar quantization. Each vector's residual is encoded
as one byte per dimension, which can reduce vector-storage memory by
about 4x vs IVF-Flat for float32 inputs (about 2x for float16 inputs),
excluding IVF structural overhead. Recall and speed trade-offs versus
IVF-PQ are dataset and tuning dependent.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `cuvs.neighbors.ivf_sq.IndexParams` |  |
| `dataset` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.ivf_sq.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_sq
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> build_params = ivf_sq.IndexParams(metric="sqeuclidean")
>>> index = ivf_sq.build(build_params, dataset)
>>> distances, neighbors = ivf_sq.search(ivf_sq.SearchParams(),
...                                      index, dataset,
...                                      k)
>>> distances = cp.asarray(distances)
>>> neighbors = cp.asarray(neighbors)
```

## extend

`@auto_sync_resources`

```python
def extend(Index index, new_vectors, new_indices, resources=None)
```

Extend an existing index with new vectors.

The input array can be either CUDA array interface compliant matrix or
array interface compliant matrix in host memory.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `ivf_sq.Index` | Trained ivf_sq object. |
| `new_vectors` | `array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16] |
| `new_indices` | `array interface compliant vector shape (n_samples)` | Supported dtype [int64] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.ivf_sq.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_sq
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> index = ivf_sq.build(ivf_sq.IndexParams(), dataset)
>>> n_rows = 100
>>> more_data = cp.random.random_sample((n_rows, n_features),
...                                     dtype=cp.float32)
>>> indices = n_samples + cp.arange(n_rows, dtype=cp.int64)
>>> index = ivf_sq.extend(index, more_data, indices)
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> distances, neighbors = ivf_sq.search(ivf_sq.SearchParams(),
...                                      index, queries,
...                                      k=10)
```

## load

`@auto_sync_resources`

```python
def load(filename, resources=None)
```

Loads index from file.

Saving / loading the index is experimental. The serialization format is
subject to change, therefore loading an index saved with a previous
version of cuvs is not guaranteed to work.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `Index` |  |

## save

`@auto_sync_resources`

```python
def save(filename, Index index, resources=None)
```

Saves the index to a file.

Saving / loading the index is experimental. The serialization format is
subject to change.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `index` | `Index` | Trained IVF-SQ index. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_sq
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = ivf_sq.build(ivf_sq.IndexParams(), dataset)
>>> # Serialize and deserialize the ivf_sq index built
>>> ivf_sq.save("my_index.bin", index)
>>> index_loaded = ivf_sq.load("my_index.bin")
```

## search

`@auto_sync_resources`
`@auto_convert_output`

```python
def search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None, filter=None)
```

Find the k nearest neighbors for each query.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `search_params` | `cuvs.neighbors.ivf_sq.SearchParams` |  |
| `index` | `cuvs.neighbors.ivf_sq.Index` | Trained IvfSq index. |
| `queries` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16] |
| `k` | `int` | The number of neighbors. |
| `neighbors` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k), dtype int64_t. If supplied, neighbor indices will be written here in-place. (default None) |
| `distances` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k) If supplied, the distances to the neighbors will be written here in-place. (default None) |
| `filter` | `Optional cuvs.neighbors.cuvsFilter can be used to filter` | neighbors based on a given bitset. (default None) |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_sq
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build the index
>>> index = ivf_sq.build(ivf_sq.IndexParams(), dataset)
>>>
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> k = 10
>>> search_params = ivf_sq.SearchParams(n_probes=20)
>>>
>>> distances, neighbors = ivf_sq.search(search_params, index, queries,
...                                     k)
```
