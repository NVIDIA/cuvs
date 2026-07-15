---
slug: api-reference/python-api-neighbors-mg-cagra
---

# Multi-GPU Cagra

_Python module: `cuvs.neighbors.mg.cagra`_

## Index

```python
cdef class Index
```

Multi-GPU CAGRA index object. Stores the trained multi-GPU CAGRA index
state which can be used to perform nearest neighbors searches across
multiple GPUs.

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
cdef class IndexParams(SingleGpuIndexParams)
```

Parameters to build multi-GPU CAGRA index for efficient search.
Extends single-GPU IndexParams with multi-GPU specific parameters.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `distribution_mode` | `str, default = "sharded"` | Distribution mode for multi-GPU setup.<br />Valid values: ["replicated", "sharded"] |
| `**kwargs` | `Additional parameters passed to single-GPU IndexParams` |  |

**Note**

CAGRA currently only supports "sqeuclidean" and "inner_product" metrics.

**Constructor**

```python
def __init__(self, *, distribution_mode="sharded", **kwargs)
```

**Members**

| Name | Kind |
| --- | --- |
| `get_handle` | method |
| `distribution_mode` | property |

### get_handle

```python
def get_handle(self)
```

### distribution_mode

```python
def distribution_mode(self)
```

## SearchParams

```python
cdef class SearchParams(SingleGpuSearchParams)
```

Parameters to search multi-GPU CAGRA index.

**Constructor**

```python
def __init__(self, *, search_mode="load_balancer", merge_mode="merge_on_root_rank", n_rows_per_batch=1000, **kwargs)
```

**Members**

| Name | Kind |
| --- | --- |
| `get_handle` | method |
| `search_mode` | property |
| `search_mode` | method |
| `merge_mode` | property |
| `merge_mode` | method |
| `n_rows_per_batch` | property |
| `n_rows_per_batch` | method |

### get_handle

```python
def get_handle(self)
```

### search_mode

```python
def search_mode(self)
```

Get the search mode for multi-GPU search.

### search_mode

```python
def search_mode(self, value)
```

Set the search mode for multi-GPU search.

### merge_mode

```python
def merge_mode(self)
```

Get the merge mode for multi-GPU search.

### merge_mode

```python
def merge_mode(self, value)
```

Set the merge mode for multi-GPU search.

### n_rows_per_batch

```python
def n_rows_per_batch(self)
```

Get the number of rows per batch for multi-GPU search.

### n_rows_per_batch

```python
def n_rows_per_batch(self, value)
```

Set the number of rows per batch for multi-GPU search.

## build

`@auto_sync_multi_gpu_resources`

```python
def build(IndexParams index_params, dataset, resources=None)
```

Build the multi-GPU CAGRA index from the dataset for efficient search.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `cuvs.neighbors.cagra.IndexParams` |  |
| `dataset` | `Array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16, int8, uint8] **IMPORTANT**: For multi-GPU CAGRA, the dataset MUST be in host memory (CPU). If using CuPy/device arrays, transfer to host with array.get() or cp.asnumpy(array). |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.cagra.Index` |  |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors.mg import cagra
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> # For multi-GPU CAGRA, use host (NumPy) arrays
>>> dataset = np.random.random_sample((n_samples, n_features)).astype(
...     np.float32)
>>> build_params = cagra.IndexParams(metric="sqeuclidean")
>>> index = cagra.build(build_params, dataset)
>>> distances, neighbors = cagra.search(cagra.SearchParams(),
...                                         index, dataset, k)
>>> # Results are already in host memory (NumPy arrays)
```

## extend

`@auto_sync_multi_gpu_resources`

```python
def extend(Index index, new_vectors, new_indices=None, resources=None)
```

Extend the multi-GPU CAGRA index with new vectors.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.cagra.Index` |  |
| `new_vectors` | `Array interface compliant matrix shape (n_new_vectors, dim)` | Supported dtype [float32, float16, int8, uint8] **IMPORTANT**: For multi-GPU CAGRA, new_vectors MUST be in host memory (CPU). If using CuPy/device arrays, transfer to host with array.get() or cp.asnumpy(array). |
| `new_indices` | `Array interface compliant matrix shape (n_new_vectors,), optional` | If provided, these indices will be used for the new vectors. If not provided, indices will be automatically assigned. **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA. Expected dtype: uint32 |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors.mg import cagra
>>> n_samples = 50000
>>> n_features = 50
>>> n_new_vectors = 1000
>>> # For multi-GPU CAGRA, use host (NumPy) arrays
>>> dataset = np.random.random_sample((n_samples, n_features)).astype(
...     np.float32)
>>> new_vectors = np.random.random_sample(
...     (n_new_vectors, n_features)).astype(np.float32)
>>> new_indices = np.arange(n_samples, n_samples + n_new_vectors,
...                         dtype=np.uint32)
>>> build_params = cagra.IndexParams(metric="sqeuclidean")
>>> index = cagra.build(build_params, dataset)
>>> cagra.extend(index, new_vectors, new_indices)  # doctest: +SKIP
```

## search

`@auto_sync_multi_gpu_resources`
`@auto_convert_output`

```python
def search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None)
```

Search the multi-GPU CAGRA index for the k-nearest neighbors of each query.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `search_params` | `cuvs.neighbors.cagra.SearchParams` |  |
| `index` | `cuvs.neighbors.cagra.Index` |  |
| `queries` | `Array interface compliant matrix shape (n_queries, dim)` | Supported dtype [float32, float16, int8, uint8] **IMPORTANT**: For multi-GPU CAGRA, queries MUST be in host memory (CPU). If using CuPy/device arrays, transfer to host with array.get() or cp.asnumpy(array). |
| `k` | `int` | The number of neighbors to search for each query. |
| `neighbors` | `Array interface compliant matrix shape (n_queries, k), optional` | If provided, this array will be filled with the indices of the k-nearest neighbors. If not provided, a new host array will be allocated. **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA. Expected dtype: int64 |
| `distances` | `Array interface compliant matrix shape (n_queries, k), optional` | If provided, this array will be filled with the distances to the k-nearest neighbors. If not provided, a new host array will be allocated. **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `distances` | `numpy.ndarray` | The distances to the k-nearest neighbors for each query (in host memory). |
| `neighbors` | `numpy.ndarray` | The indices of the k-nearest neighbors for each query (in host memory). |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors.mg import cagra
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> # For multi-GPU CAGRA, use host (NumPy) arrays
>>> dataset = np.random.random_sample((n_samples, n_features)).astype(
...     np.float32)
>>> queries = np.random.random_sample((n_queries, n_features)).astype(
...     np.float32)
>>> build_params = cagra.IndexParams(metric="sqeuclidean")
>>> index = cagra.build(build_params, dataset)
>>> distances, neighbors = cagra.search(cagra.SearchParams(),
...                                         index, queries, k)
>>> # Results are already in host memory (NumPy arrays)
```

## save

`@auto_sync_multi_gpu_resources`

```python
def save(Index index, filename, resources=None)
```

Serialize the multi-GPU CAGRA index to a file.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.cagra.Index` |  |
| `filename` | `str` | The filename to serialize the index to. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors.mg import cagra
>>> n_samples = 50000
>>> n_features = 50
>>> # For multi-GPU CAGRA, use host (NumPy) arrays
>>> dataset = np.random.random_sample((n_samples, n_features)).astype(
...     np.float32)
>>> build_params = cagra.IndexParams(metric="sqeuclidean")
>>> index = cagra.build(build_params, dataset)
>>> cagra.save(index, "index.bin")
```

## load

`@auto_sync_multi_gpu_resources`

```python
def load(filename, resources=None)
```

Deserialize the multi-GPU CAGRA index from a file.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `str` | The filename to deserialize the index from. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `Index` | The deserialized index. |

**Examples**

```python
>>> from cuvs.neighbors.mg import cagra
>>> index = cagra.load("index.bin")  # doctest: +SKIP
```

## distribute

`@auto_sync_multi_gpu_resources`

```python
def distribute(filename, resources=None)
```

Distribute a single-GPU CAGRA index across multiple GPUs from a file.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `str` | The filename to distribute the index from. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `Index` | The distributed index. |

**Examples**

```python
>>> from cuvs.neighbors.mg import cagra
>>> index = cagra.distribute("single_gpu_index.bin")  # doctest: +SKIP
```
