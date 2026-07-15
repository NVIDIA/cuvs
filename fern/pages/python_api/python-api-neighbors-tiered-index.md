---
slug: api-reference/python-api-neighbors-tiered-index
---

# Tiered Index

_Python module: `cuvs.neighbors.tiered_index`_

## Index

```python
cdef class Index
```

Tiered Index object.

**Members**

| Name | Kind |
| --- | --- |
| `trained` | property |

### trained

```python
def trained(self)
```

## IndexParams

```python
cdef class IndexParams
```

Parameters to build index for Tiered Index nearest neighbor search

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `str, default = "sqeuclidean"` | String denoting the metric type.<br />Valid values for metric: ["sqeuclidean", "inner_product", "euclidean", "cosine"], where<br />- sqeuclidean is the euclidean distance without the square root operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,<br />- euclidean is the euclidean distance<br />- inner product distance is defined as distance(a, b) = \\sum_i a_i * b_i.<br />- cosine distance is defined as distance(a, b) = 1 - \\sum_i a_i * b_i / ( \|\|a\|\|_2 * \|\|b\|\|_2). |
| `algo` | `str, default = "cagra"` | The algorithm to use for the ANN portion of the tiered index |
| `upstream_params` | `object, optional` | The IndexParams for the upstream ANN object to use (ie the Cagra IndexParams for cagra etc) |
| `min_ann_rows` | `int` | The minimum number of rows necessary to create an ann index |
| `create_ann_index_on_extend` | `bool` | Whether or not to create a new ann index on extend, if the number of rows in the incremental (bfknn) portion is above min_ann_rows |

**Constructor**

```python
def __init__(self, *, metric="sqeuclidean", algo="cagra", upstream_params=None, min_ann_rows=None, create_ann_index_on_extend=None,)
```

**Members**

| Name | Kind |
| --- | --- |
| `metric` | property |
| `algo` | property |
| `min_ann_rows` | property |
| `create_ann_index_on_extend` | property |
| `upstream_params` | property |

### metric

```python
def metric(self)
```

### algo

```python
def algo(self)
```

### min_ann_rows

```python
def min_ann_rows(self)
```

### create_ann_index_on_extend

```python
def create_ann_index_on_extend(self)
```

### upstream_params

```python
def upstream_params(self)
```

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, resources=None)
```

Build the Tiered index from the dataset for efficient search.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `cuvs.neighbors.tiered_index.IndexParams` |  |
| `dataset` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.tiered_index.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import cagra, tiered_index
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> build_params = tiered_index.IndexParams(metric="sqeuclidean",
...                                         algo="cagra")
>>> index = tiered_index.build(build_params, dataset)
>>> distances, neighbors = tiered_index.search(cagra.SearchParams(),
...                                            index, dataset, k)
>>> distances = cp.asarray(distances)
>>> neighbors = cp.asarray(neighbors)
```

## extend

`@auto_sync_resources`

```python
def extend(Index index, new_vectors, resources=None)
```

Extend an existing index with new vectors.

The input array can be either CUDA array interface compliant matrix or
array interface compliant matrix in host memory.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `tiered_index.Index` | Trained tiered_index object. |
| `new_vectors` | `array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.tiered_index.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import tiered_index
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> index = tiered_index.build(tiered_index.IndexParams(), dataset)
>>> n_rows = 100
>>> more_data = cp.random.random_sample((n_rows, n_features),
...                                     dtype=cp.float32)
>>> index = tiered_index.extend(index, more_data)
```

## search

`@auto_sync_resources`
`@auto_convert_output`

```python
def search(search_params, Index index, queries, k, neighbors=None, distances=None, resources=None, filter=None)
```

Find the k nearest neighbors for each query.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `search_params` | `SearchParams for the upstream ANN index` |  |
| `index` | `cuvs.neighbors.tiered_index.Index` | Trained Tiered index. |
| `queries` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32] |
| `k` | `int` | The number of neighbors. |
| `neighbors` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k), dtype int64_t. If supplied, neighbor indices will be written here in-place. (default None) |
| `distances` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k) If supplied, the distances to the neighbors will be written here in-place. (default None) |
| `filter` | `Optional cuvs.neighbors.cuvsFilter can be used to filter` | neighbors based on a given bitset. (default None) |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import cagra, tiered_index
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build the index
>>> index = tiered_index.build(tiered_index.IndexParams(algo="cagra"),
...                            dataset)
>>>
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> k = 10
>>> search_params = cagra.SearchParams()
>>>
>>> distances, neighbors = tiered_index.search(search_params, index,
...                                            queries, k)
```
