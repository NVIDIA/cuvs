---
slug: api-reference/python-api-neighbors-brute-force
---

# Brute Force

_Python module: `cuvs.neighbors.brute_force`_

## Index

```python
cdef class Index
```

Brute Force index object. This object stores the trained Brute Force
which can be used to perform nearest neighbors searches.

**Members**

| Name | Kind |
| --- | --- |
| `trained` | property |

### trained

```python
def trained(self)
```

## build

`@auto_sync_resources`

```python
def build(dataset, metric="sqeuclidean", metric_arg=2.0, resources=None)
```

Build the Brute Force index from the dataset for efficient search.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dataset` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16] |
| `metric` | `Distance metric to use. Default is sqeuclidean` |  |
| `metric_arg` | `value of 'p' for Minkowski distances` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.brute_force.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import brute_force
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> index = brute_force.build(dataset, metric="cosine")
>>> distances, neighbors = brute_force.search(index, dataset, k)
>>> distances = cp.asarray(distances)
>>> neighbors = cp.asarray(neighbors)
```

## search

`@auto_sync_resources`
`@auto_convert_output`

```python
def search(Index index, queries, k, neighbors=None, distances=None, resources=None, prefilter=None)
```

Find the k nearest neighbors for each query.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `Index` | Trained Brute Force index. |
| `queries` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16] |
| `k` | `int` | The number of neighbors. |
| `neighbors` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k), dtype int64_t. If supplied, neighbor indices will be written here in-place. (default None) |
| `distances` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k) If supplied, the distances to the neighbors will be written here in-place. (default None) |
| `prefilter` | `Optional, cuvs.neighbors.cuvsFilter` | An optional filter to exclude certain query-neighbor pairs using a bitmap or bitset. The filter function should have a row-major layout with logical shape `(n_prefilter_rows, n_samples)`, where:<br />- `n_prefilter_rows == n_queries` when using a bitmap filter.<br />- `n_prefilter_rows == 1` when using a bitset prefilter. Each bit in `n_samples` determines whether `queries[i]` should be considered for distance computation with the index. (default None) |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> # Example without pre-filter
>>> import cupy as cp
>>> from cuvs.neighbors import brute_force
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = brute_force.build(dataset, metric="sqeuclidean")
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> k = 10
>>> # Using a pooling allocator reduces overhead of temporary array
>>> # creation during search. This is useful if multiple searches
>>> # are performed with same query size.
>>> distances, neighbors = brute_force.search(index, queries, k)
>>> neighbors = cp.asarray(neighbors)
>>> distances = cp.asarray(distances)
```

```python
>>> # Example with pre-filter
>>> import numpy as np
>>> import cupy as cp
>>> from cuvs.neighbors import brute_force, filters
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = brute_force.build(dataset, metric="sqeuclidean")
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> # Build filters
>>> n_bitmap = np.ceil(n_samples * n_queries / 32).astype(int)
>>> # Create your own bitmap as the filter by replacing the random one.
>>> bitmap = cp.random.randint(1, 100, size=(n_bitmap,), dtype=cp.uint32)
>>> bitmap_prefilter = filters.from_bitmap(bitmap)
>>>
>>> # or Build bitset prefilter:
>>> # n_bitset = np.ceil(n_samples * 1 / 32).astype(int)
>>> # # Create your own bitset as the filter by replacing the random one.
>>> # bitset = cp.random.randint(1, 100, size=(n_bitset,), dtype=cp.uint32)
>>> # bitset_prefilter = filters.from_bitset(bitset)
>>>
>>> k = 10
>>> # Using a pooling allocator reduces overhead of temporary array
>>> # creation during search. This is useful if multiple searches
>>> # are performed with same query size.
>>> distances, neighbors = brute_force.search(index, queries, k,
...                                           prefilter=bitmap_prefilter)
>>> neighbors = cp.asarray(neighbors)
>>> distances = cp.asarray(distances)
```

## save

`@auto_sync_resources`

```python
def save(filename, Index index, bool include_dataset=True, resources=None)
```

Saves the index to a file.

The serialization format can be subject to changes, therefore loading
an index saved with a previous version of cuvs is not guaranteed
to work.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `index` | `Index` | Trained Brute Force index. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import brute_force
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = brute_force.build(dataset)
>>> # Serialize and deserialize the brute_force index built
>>> brute_force.save("my_index.bin", index)
>>> index_loaded = brute_force.load("my_index.bin")
```

## load

`@auto_sync_resources`

```python
def load(filename, resources=None)
```

Loads index from file.

The serialization format can be subject to changes, therefore loading
an index saved with a previous version of cuvs is not guaranteed
to work.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import brute_force
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = brute_force.build(dataset)
>>> # Serialize and deserialize the brute_force index built
>>> brute_force.save("my_index.bin", index)
>>> index_loaded = brute_force.load("my_index.bin")
```
