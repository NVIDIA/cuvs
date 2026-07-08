---
slug: api-reference/python-api-neighbors-hnsw
---

# HNSW

_Python module: `cuvs.neighbors.hnsw`_

## AceParams

```python
cdef class AceParams
```

Parameters for ACE (Augmented Core Extraction) graph build for HNSW.

ACE enables building HNSW indices for datasets too large to fit in GPU
memory by partitioning the dataset and building sub-indices for each
partition independently.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `npartitions` | `int, default = 0 (optional)` | Number of partitions for ACE partitioned build. When set to 0 (default), the number of partitions is automatically derived based on available host and GPU memory to maximize partition size while ensuring the build fits in memory.<br /><br />Small values might improve recall but potentially degrade performance and increase memory usage. Partitions should not be too small to prevent issues in KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests).<br /><br />If the specified number of partitions results in partitions that exceed available memory, the value will be automatically increased to fit memory constraints and a warning will be issued. |
| `build_dir` | `string, default = "/tmp/hnsw_ace_build" (optional)` | Directory to store ACE build artifacts (KNN graph, optimized graph). Used when `use_disk` is true or when the graph does not fit in memory. |
| `use_disk` | `bool, default = False (optional)` | Whether to use disk-based storage for ACE build. When true, enables disk-based operations for memory-efficient graph construction. |
| `max_host_memory_gb` | `float, default = 0 (optional)` | Maximum host memory to use for ACE build in GiB. When set to 0 (default), uses available host memory. Useful for testing or when running alongside other memory-intensive processes. |
| `max_gpu_memory_gb` | `float, default = 0 (optional)` | Maximum GPU memory to use for ACE build in GiB. When set to 0 (default), uses available GPU memory. Useful for testing or when running alongside other memory-intensive processes. |

**Constructor**

```python
def __init__(self, *, npartitions=0, build_dir="/tmp/hnsw_ace_build", use_disk=False, max_host_memory_gb=0, max_gpu_memory_gb=0)
```

**Members**

| Name | Kind |
| --- | --- |
| `npartitions` | property |
| `build_dir` | property |
| `use_disk` | property |
| `max_host_memory_gb` | property |
| `max_gpu_memory_gb` | property |

### npartitions

```python
def npartitions(self)
```

### build_dir

```python
def build_dir(self)
```

### use_disk

```python
def use_disk(self)
```

### max_host_memory_gb

```python
def max_host_memory_gb(self)
```

### max_gpu_memory_gb

```python
def max_gpu_memory_gb(self)
```

## IndexParams

```python
cdef class IndexParams
```

Parameters to build index for HNSW nearest neighbor search

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `hierarchy` | `string, default = "gpu" (optional)` | The hierarchy of the HNSW index.<br />Valid values are ["none", "cpu", "gpu"].<br />- "none": No hierarchy is built.<br />- "cpu": Hierarchy is built using CPU.<br />- "gpu": Hierarchy is built using GPU. |
| `ef_construction` | `int, default = 200 (optional)` | Maximum candidate list size used during index construction. |
| `num_threads` | `int, default = 0 (optional)` | Number of CPU threads used to increase construction parallelism when hierarchy is `cpu` or `gpu`. When the value is 0, the number of threads is automatically determined to the maximum number of threads available.<br />NOTE: When hierarchy is `gpu`, while the majority of the work is done on the GPU, initialization of the HNSW index itself and some other work is parallelized with the help of CPU threads. |
| `M` | `int, default = 32 (optional)` | HNSW M parameter: number of bi-directional links per node used to derive the internal GPU graph degree. |
| `metric` | `string, default = "sqeuclidean" (optional)` | Distance metric to use.<br />Valid values: ["sqeuclidean", "inner_product"] |
| `ace_params` | `AceParams, default = None (optional)` | Optional ACE parameters for partitioned or disk-backed GPU graph construction. If not set, build() uses HNSW parameters and selects internal graph build settings automatically. |

**Constructor**

```python
def __init__(self, *, hierarchy="gpu", ef_construction=200, num_threads=0, M=32, metric="sqeuclidean", ace_params=None)
```

**Members**

| Name | Kind |
| --- | --- |
| `hierarchy` | property |
| `ef_construction` | property |
| `num_threads` | property |
| `m` | property |
| `metric` | property |
| `ace_params` | property |

### hierarchy

```python
def hierarchy(self)
```

### ef_construction

```python
def ef_construction(self)
```

### num_threads

```python
def num_threads(self)
```

### m

```python
def m(self)
```

### metric

```python
def metric(self)
```

### ace_params

```python
def ace_params(self)
```

## Index

```python
cdef class Index
```

HNSW index object. This object stores the trained HNSW index state
which can be used to perform nearest neighbors searches.

**Members**

| Name | Kind |
| --- | --- |
| `trained` | property |

### trained

```python
def trained(self)
```

## ExtendParams

```python
cdef class ExtendParams
```

Parameters to extend the HNSW index with new data

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `num_threads` | `int, default = 0 (optional)` | Number of CPU threads used to increase construction parallelism. When set to 0, the number of threads is automatically determined. |

**Constructor**

```python
def __init__(self, *, num_threads=0)
```

**Members**

| Name | Kind |
| --- | --- |
| `num_threads` | property |

### num_threads

```python
def num_threads(self)
```

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, resources=None)
```

Build an HNSW index on the GPU and search it on the CPU.

The build API accepts HNSW parameters (`M`, `ef_construction`,
`hierarchy`, and `metric`) and selects the internal GPU graph
construction settings automatically. Set `ace_params` only when
partitioned or disk-backed graph construction needs to be configured.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `IndexParams` | Parameters for the HNSW index. |
| `dataset` | `Host array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16, int8, uint8] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `Index` | Trained HNSW index ready for search. |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors import hnsw
>>>
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = np.random.random_sample(
...     (n_samples, n_features)).astype(np.float32)
>>>
>>> # Create HNSW index parameters
>>> index_params = hnsw.IndexParams(
...     hierarchy="gpu",
...     ef_construction=120,
...     M=32,
...     metric="sqeuclidean"
... )
>>>
>>> # Build the index
>>> index = hnsw.build(index_params, dataset)
>>>
>>> # Search the index
>>> queries = np.random.random_sample((10, n_features)).astype(np.float32)
>>> distances, neighbors = hnsw.search(
...     hnsw.SearchParams(ef=200),
...     index,
...     queries,
...     k=10
... )
```

## extend

`@auto_sync_resources`

```python
def extend(ExtendParams extend_params, Index index, data, resources=None)
```

Extends the HNSW index with new data.

Indexes with hierarchy `"cpu"` or `"gpu"` can be extended. A
base-layer-only index with hierarchy `"none"` cannot be extended.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `extend_params` | `ExtendParams` |  |
| `index` | `Index` | Trained HNSW index. |
| `data` | `Host array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16, int8, uint8] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors import hnsw
>>>
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = np.random.random_sample(
...     (n_samples, n_features)).astype(np.float32)
>>> hnsw_index = hnsw.build(hnsw.IndexParams(), dataset)
>>> # Extend the index with new data
>>> new_data = np.random.random_sample(
...     (n_samples, n_features)).astype(np.float32)
>>> hnsw.extend(hnsw.ExtendParams(), hnsw_index, new_data)
```

## SearchParams

```python
cdef class SearchParams
```

HNSW search parameters

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `ef` | `int, default = 200` | Maximum number of candidate list size used during search. Must not be negative. When 0, the effective candidate list size is k. |
| `num_threads` | `int, default = 0` | Number of CPU threads used to increase search parallelism. When set to 0, the number of threads is automatically determined using OpenMP's `omp_get_max_threads()`. |

**Constructor**

```python
def __init__(self, *, ef=200, num_threads=0)
```

**Members**

| Name | Kind |
| --- | --- |
| `ef` | property |
| `num_threads` | property |

### ef

```python
def ef(self)
```

### num_threads

```python
def num_threads(self)
```

## load

`@auto_sync_resources`

```python
def load(IndexParams index_params, filename, dim, dtype, metric=None, resources=None)
```

Loads an HNSW index.
If the index was constructed with `hnsw.IndexParams(hierarchy="none")`,
then the loaded index is immutable and can only be searched by the hnswlib
wrapper in cuVS, as the format is not compatible with the original hnswlib.
However, if the index was constructed with hierarchy `"cpu"` or `"gpu"`,
then the loaded index is mutable and compatible with the original hnswlib.

Saving / loading the index is experimental. The serialization format is
subject to change, therefore loading an index saved with a previous
version of cuVS is not guaranteed to work.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `IndexParams` | Parameters that were used to build or convert the HNSW index. `index_params.hierarchy` must match the hierarchy the index was saved with (e.g. `"none"` for a base-layer-only index). |
| `filename` | `string` | Name of the file. |
| `dim` | `int` | Dimensions of the training dataset |
| `dtype` | `np.dtype of the saved index` | Valid values for dtype: [np.float32, np.float16, np.byte, np.ubyte] |
| `metric` | `string denoting the metric type, default=None` | When None, the metric is taken from `index_params`. If provided, it takes precedence over `index_params.metric`.<br />Valid values for metric: ["sqeuclidean", "inner_product"], where<br />- sqeuclidean is the euclidean distance without the square root operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,<br />- inner_product distance is defined as distance(a, b) = \\sum_i a_i * b_i. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `HnswIndex` |  |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors import hnsw
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = np.random.random_sample(
...     (n_samples, n_features)).astype(np.float32)
>>> index_params = hnsw.IndexParams(metric="sqeuclidean")
>>> index = hnsw.build(index_params, dataset)
>>> hnsw.save("my_index.bin", index)
>>> index = hnsw.load(index_params, "my_index.bin", n_features,
...                   np.float32, "sqeuclidean")
```

## save

`@auto_sync_resources`

```python
def save(filename, Index index, resources=None)
```

Saves an HNSW index to a file.
If the index was constructed with `hnsw.IndexParams(hierarchy="none")`,
then the saved index is immutable and can only be searched by the hnswlib
wrapper in cuVS, as the format is not compatible with the original hnswlib.
However, if the index was constructed with hierarchy `"cpu"` or `"gpu"`,
then the saved index is mutable and compatible with the original hnswlib.

Saving / loading the index is experimental. The serialization format is
subject to change.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `index` | `Index` | Trained HNSW index. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors import hnsw
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = np.random.random_sample(
...     (n_samples, n_features)).astype(np.float32)
>>> hnsw_index = hnsw.build(hnsw.IndexParams(), dataset)
>>> hnsw.save("my_index.bin", hnsw_index)
```

## search

`@auto_sync_resources`
`@auto_convert_output`

```python
def search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None)
```

Find the k nearest neighbors for each query.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `search_params` | `SearchParams` |  |
| `index` | `Index` | Trained HNSW index. |
| `queries` | `CPU array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float, int] |
| `k` | `int` | The number of neighbors. |
| `neighbors` | `Optional CPU array interface compliant matrix shape` | (n_queries, k), dtype uint64_t. If supplied, neighbor indices will be written here in-place. (default None) |
| `distances` | `Optional CPU array interface compliant matrix shape` | (n_queries, k) If supplied, the distances to the neighbors will be written here in-place. (default None) |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import numpy as np
>>> from cuvs.neighbors import hnsw
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = np.random.random_sample(
...     (n_samples, n_features)).astype(np.float32)
>>> index = hnsw.build(hnsw.IndexParams(), dataset)
>>> queries = np.random.random_sample(
...     (n_queries, n_features)).astype(np.float32)
>>> k = 10
>>> search_params = hnsw.SearchParams(
...     ef=200,
...     num_threads=0
... )
>>> distances, neighbors = hnsw.search(search_params, index, queries, k)
```

## from_cagra

`@auto_sync_resources`

```python
def from_cagra(IndexParams index_params, cagra.Index cagra_index, temporary_index_path=None, resources=None)
```

Returns an HNSW index from a CAGRA index.

NOTE: When `index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in
`/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then
deleting the temporary file. The returned index is immutable and can only
be searched by the hnswlib wrapper in cuVS, as the format is not
compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional
vectors. The serialized index is also compatible with the original hnswlib
library.
3. `GPU`: The returned index is mutable, and its hierarchy is built on the
GPU. The serialized index is also compatible with the original hnswlib
library.

Saving / loading the index is experimental. The serialization format is
subject to change.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `IndexParams` | Parameters to convert the CAGRA index to HNSW index. |
| `cagra_index` | `cagra.Index` | Trained CAGRA index. |
| `temporary_index_path` | `string, default = None` | Deprecated and ignored. The temporary file used when hierarchy is `NONE` is always written to the system temporary directory. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import cagra
>>> from cuvs.neighbors import hnsw
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = cagra.build(cagra.IndexParams(), dataset)
>>> # Convert the CAGRA index to an HNSW index
>>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), index)
```
