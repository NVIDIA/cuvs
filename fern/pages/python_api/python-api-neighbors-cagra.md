---
slug: api-reference/python-api-neighbors-cagra
---

# Cagra

_Python module: `cuvs.neighbors.cagra`_

## AceParams

```python
cdef class AceParams
```

Parameters for ACE (Augmented Core Extraction) graph building algorithm.

ACE enables building indexes for datasets too large to fit in GPU memory by
partitioning the dataset using balanced k-means and building sub-indexes
for each partition independently.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `npartitions` | `int, default = 0` | Number of partitions for ACE partitioned build. When set to 0 (default), the number of partitions is automatically derived based on available host and GPU memory to maximize partition size while ensuring the build fits in memory.<br /><br />Small values might improve recall but potentially degrade performance and increase memory usage. Partitions should not be too small to prevent issues in KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests).<br /><br />If the specified number of partitions results in partitions that exceed available memory, the value will be automatically increased to fit memory constraints and a warning will be issued. |
| `ef_construction` | `int, default = 120` | The index quality for the ACE build. Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| `build_dir` | `str, default = "/tmp/ace_build"` | Directory to store ACE build artifacts (e.g., KNN graph, optimized graph). Used when `use_disk` is true or when the graph does not fit in host and GPU memory. This should be the fastest disk in the system and hold enough space for twice the dataset, final graph, and label mapping. |
| `use_disk` | `bool, default = False` | Whether to use disk-based storage for ACE build. When true, enables disk-based operations for memory-efficient graph construction. |
| `max_host_memory_gb` | `float, default = 0` | Maximum host memory to use for ACE build in GiB. When set to 0 (default), uses available host memory. Useful for testing or when running alongside other memory-intensive processes. |
| `max_gpu_memory_gb` | `float, default = 0` | Maximum GPU memory to use for ACE build in GiB. When set to 0 (default), uses available GPU memory. Useful for testing or when running alongside other memory-intensive processes. |

**Constructor**

```python
def __init__(self, *, npartitions=0, ef_construction=120, build_dir="/tmp/ace_build", use_disk=False, max_host_memory_gb=0, max_gpu_memory_gb=0)
```

**Members**

| Name | Kind |
| --- | --- |
| `npartitions` | property |
| `ef_construction` | property |
| `build_dir` | property |
| `use_disk` | property |
| `max_host_memory_gb` | property |
| `max_gpu_memory_gb` | property |
| `get_handle` | method |

### npartitions

```python
def npartitions(self)
```

### ef_construction

```python
def ef_construction(self)
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

### get_handle

```python
def get_handle(self)
```

## CompressionParams

```python
cdef class CompressionParams
```

Parameters for VPQ Compression

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `int` | The bit length of the vector element after compression by PQ.<br />Possible values: [4, 5, 6, 7, 8]. The smaller the 'pq_bits', the smaller the index size and the better the search performance, but the lower the recall. |
| `pq_dim` | `int` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is selected using a heuristic. |
| `vq_n_centers` | `int` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". When zero, an optimal value is selected using a heuristic. |
| `kmeans_n_iters` | `int` | The number of iterations searching for kmeans centers (both VQ & PQ phases). |
| `vq_kmeans_trainset_fraction` | `float` | The fraction of data to use during iterative kmeans building (VQ phase). When zero, an optimal value is selected using a heuristic. |
| `pq_kmeans_trainset_fraction` | `float` | The fraction of data to use during iterative kmeans building (PQ phase). When zero, an optimal value is selected using a heuristic. |

**Constructor**

```python
def __init__(self, *, pq_bits=8, pq_dim=0, vq_n_centers=0, kmeans_n_iters=25, vq_kmeans_trainset_fraction=0.0, pq_kmeans_trainset_fraction=0.0)
```

**Members**

| Name | Kind |
| --- | --- |
| `pq_bits` | property |
| `pq_dim` | property |
| `vq_n_centers` | property |
| `kmeans_n_iters` | property |
| `vq_kmeans_trainset_fraction` | property |
| `pq_kmeans_trainset_fraction` | property |
| `get_handle` | method |

### pq_bits

```python
def pq_bits(self)
```

### pq_dim

```python
def pq_dim(self)
```

### vq_n_centers

```python
def vq_n_centers(self)
```

### kmeans_n_iters

```python
def kmeans_n_iters(self)
```

### vq_kmeans_trainset_fraction

```python
def vq_kmeans_trainset_fraction(self)
```

### pq_kmeans_trainset_fraction

```python
def pq_kmeans_trainset_fraction(self)
```

### get_handle

```python
def get_handle(self)
```

## ExtendParams

```python
cdef class ExtendParams
```

Supplemental parameters to extend CAGRA Index

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `max_chunk_size` | `int` | The additional dataset is divided into chunks and added to the graph. This is the knob to adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also degrade recall because no edges are added between the nodes in the same chunk. Auto select when 0. |

**Constructor**

```python
def __init__(self, *, max_chunk_size=None)
```

**Members**

| Name | Kind |
| --- | --- |
| `max_chunk_size` | property |

### max_chunk_size

```python
def max_chunk_size(self)
```

## Index

```python
cdef class Index
```

**Members**

| Name | Kind |
| --- | --- |
| `trained` | property |
| `dim` | property |
| `graph_degree` | property |
| `dtype` | property |
| `dataset` | property |
| `graph` | property |

### trained

```python
def trained(self)
```

### dim

```python
def dim(self)
```

### graph_degree

```python
def graph_degree(self)
```

### dtype

```python
def dtype(self)
```

### dataset

```python
def dataset(self)
```

### graph

```python
def graph(self)
```

## IndexParams

```python
cdef class IndexParams
```

Parameters to build index for CAGRA nearest neighbor search

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `str, default = "sqeuclidean"` | String denoting the metric type,<br />valid values for metric are ["sqeuclidean", "inner_product", "cosine"], where:<br /><br />- sqeuclidean is the euclidean distance without the square root operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2<br />- inner_product distance is defined as distance(a, b) = \\sum_i a_i * b_i.<br />- cosine distance is defined as distance(a, b) = 1 - \\sum_i a_i * b_i / ( \|\|a\|\|_2 * \|\|b\|\|_2). |
| `intermediate_graph_degree` | `int, default = 128` |  |
| `graph_degree` | `int, default = 64` |  |
| `build_algo` | `str, default = "ivf_pq"` | string denoting the graph building algorithm to use.<br />Valid values for algo: ["ivf_pq", "nn_descent", "iterative_cagra_search", "ace"], where<br /><br />- ivf_pq will use the IVF-PQ algorithm for building the knn graph<br />- nn_descent (experimental) will use the NN-Descent algorithm for building the knn graph. It is expected to be generally faster than ivf_pq.<br />- iterative_cagra_search will iteratively build the knn graph using CAGRA's search() and optimize()<br />- ace will use ACE (Augmented Core Extraction) for building indices for datasets too large to fit in GPU memory |
| `compression` | `CompressionParams, optional` | If compression is desired should be a CompressionParams object. If None compression will be disabled. |
| `ivf_pq_build_params` | `cuvs.neighbors.ivf_pq.IndexParams, optional` | Parameters for IVF-PQ algorithm. If provided, it will be used for building the graph. |
| `ivf_pq_search_params` | `cuvs.neighbors.ivf_pq.SearchParams, optional` | Parameters for IVF-PQ search. If provided, it will be used for searching the graph. |
| `ace_params` | `AceParams, optional` | Parameters for ACE algorithm. If provided, it will be used for building the graph with ACE partitioning. |
| `refinement_rate` | `float, default = 1.0` |  |

**Constructor**

```python
def __init__(self, *, metric="sqeuclidean", intermediate_graph_degree=128, graph_degree=64, build_algo="ivf_pq", nn_descent_niter=20, compression=None, ivf_pq_build_params: ivf_pq.IndexParams = None, ivf_pq_search_params: ivf_pq.SearchParams = None, ace_params: AceParams = None, refinement_rate: float = 1.0)
```

**Members**

| Name | Kind |
| --- | --- |
| `get_handle` | method |
| `metric` | property |
| `intermediate_graph_degree` | property |
| `graph_degree` | property |
| `build_algo` | property |
| `nn_descent_niter` | property |
| `refinement_rate` | property |

### get_handle

```python
def get_handle(self)
```

### metric

```python
def metric(self)
```

### intermediate_graph_degree

```python
def intermediate_graph_degree(self)
```

### graph_degree

```python
def graph_degree(self)
```

### build_algo

```python
def build_algo(self)
```

### nn_descent_niter

```python
def nn_descent_niter(self)
```

### refinement_rate

```python
def refinement_rate(self)
```

## SearchParams

```python
cdef class SearchParams
```

CAGRA search parameters

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `max_queries` | `int, default = 0` | Maximum number of queries to search at the same time (batch size). Auto select when 0. |
| `itopk_size` | `int, default = 64` | Number of intermediate search results retained during the search. This is the main knob to adjust trade off between accuracy and search speed. Higher values improve the search accuracy. |
| `max_iterations` | `int, default = 0` | Upper limit of search iterations. Auto select when 0. |
| `algo` | `str, default = "auto"` | String denoting the search algorithm to use<br />Valid values for algo: ["auto", "single_cta", "multi_cta"], where:<br /><br />- auto will automatically select the best value based on query size<br />- single_cta is better when query contains larger number of vectors (e.g &gt;10)<br />- multi_cta is better when query contains only a few vectors |
| `team_size` | `int, default = 0` | Number of threads used to calculate a single distance. 4, 8, 16, or 32. |
| `search_width` | `int, default = 1` | Number of graph nodes to select as the starting point for the search in each iteration. |
| `min_iterations` | `int, default = 0` | Lower limit of search iterations. |
| `thread_block_size` | `int, default = 0` | Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. |
| `hashmap_mode` | `str, default = "auto"` | String denoting the type of hash map to use. It's usually better to allow the algorithm to select this value,<br />Valid values for hashmap_mode: ["auto", "small", "hash"], where:<br /><br />- auto will automatically select the best value based on algo<br />- small will use the small shared memory hash table with resetting.<br />- hash will use a single hash table in global memory. |
| `hashmap_min_bitlen` | `int, default = 0` | Upper limit of hashmap fill rate. More than 0.1, less than 0.9. |
| `hashmap_max_fill_rate` | `float, default = 0.5` | Upper limit of hashmap fill rate. More than 0.1, less than 0.9. |
| `num_random_samplings` | `int, default = 1` | Number of iterations of initial random seed node selection. 1 or more. |
| `rand_xor_mask` | `int, default = 0x128394` | Bit mask used for initial random seed node selection. |
| `persistent` | `bool, default = false` | Whether to use the persistent version of the kernel |
| `persistent_lifetime` | `float` | Persistent kernel: time in seconds before the kernel stops if no requests are received. |
| `persistent_device_usage` | `float` | Sets the fraction of maximum grid size used by persistent kernel. |

**Constructor**

```python
def __init__(self, *, max_queries=0, itopk_size=64, max_iterations=0, algo="auto", team_size=0, search_width=1, min_iterations=0, thread_block_size=0, hashmap_mode="auto", hashmap_min_bitlen=0, hashmap_max_fill_rate=0.5, num_random_samplings=1, rand_xor_mask=0x128394, persistent=False, persistent_lifetime=None, persistent_device_usage=None )
```

**Members**

| Name | Kind |
| --- | --- |
| `get_handle` | method |
| `max_queries` | property |
| `itopk_size` | property |
| `max_iterations` | property |
| `algo` | property |
| `team_size` | property |
| `search_width` | property |
| `min_iterations` | property |
| `thread_block_size` | property |
| `hashmap_mode` | property |
| `hashmap_min_bitlen` | property |
| `hashmap_max_fill_rate` | property |
| `num_random_samplings` | property |
| `rand_xor_mask` | property |

### get_handle

```python
def get_handle(self)
```

### max_queries

```python
def max_queries(self)
```

### itopk_size

```python
def itopk_size(self)
```

### max_iterations

```python
def max_iterations(self)
```

### algo

```python
def algo(self)
```

### team_size

```python
def team_size(self)
```

### search_width

```python
def search_width(self)
```

### min_iterations

```python
def min_iterations(self)
```

### thread_block_size

```python
def thread_block_size(self)
```

### hashmap_mode

```python
def hashmap_mode(self)
```

### hashmap_min_bitlen

```python
def hashmap_min_bitlen(self)
```

### hashmap_max_fill_rate

```python
def hashmap_max_fill_rate(self)
```

### num_random_samplings

```python
def num_random_samplings(self)
```

### rand_xor_mask

```python
def rand_xor_mask(self)
```

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, resources=None)
```

Build the CAGRA index from the dataset for efficient search.

The build performs two different steps- first an intermediate knn-graph is
constructed, then it's optimized it to create the final graph. The
index_params object controls the node degree of these graphs.

It is required that both the dataset and the optimized graph fit the
GPU memory.

Note: When using ACE (Augmented Core Extraction) build algorithm, the
dataset must be in host memory (CPU). The ACE algorithm is designed for
datasets too large to fit in GPU memory.

The following distance metrics are supported:
- L2
- InnerProduct
- Cosine

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `IndexParams object` |  |
| `dataset` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float, half, int8, uint8] **Note:** For ACE build algorithm, the dataset MUST be in host memory. Use NumPy arrays or call .get() on CuPy arrays before passing. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.cagra.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import cagra
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> build_params = cagra.IndexParams(metric="sqeuclidean")
>>> index = cagra.build(build_params, dataset)
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> distances, neighbors = cagra.search(cagra.SearchParams(),
...                                      index, queries,
...                                      k)
>>> distances = cp.asarray(distances)
>>> neighbors = cp.asarray(neighbors)
```

## extend

`@auto_sync_resources`

```python
def extend(ExtendParams params, Index index, additional_dataset, resources=None)
```

Extend a CAGRA index with additional vectors

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `ExtendParams object` |  |
| `index` | `Index` | Existing cagra index to extend |
| `additional_dataset` | `CUDA array interface compliant matrix shape` | Supported dtype [float, half, int8, uint8] |
| `resources` | `cuvs.common.Resources, optional` |  |

## from_graph

`@auto_sync_resources`

```python
def from_graph(graph, dataset, metric="sqeuclidean", resources=None)
```

Construct a cagra index from an existing graph and dataset

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `graph` | `Array interface compliant matrix with shape (n_samples, graph_degree)` |  |
| `dataset` | `Array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16, int8, uint8] |
| `metric` | `str` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.cagra.Index` |  |

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
def save(filename, Index index, bool include_dataset=True, resources=None)
```

Saves the index to a file.

Saving / loading the index is experimental. The serialization format is
subject to change.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `index` | `Index` | Trained CAGRA index. |
| `include_dataset` | `bool` | Whether or not to write out the dataset along with the index. Including the dataset in the serialized index will use extra disk space, and might not be desired if you already have a copy of the dataset on disk. If this option is set to false, you will have to call `index.update_dataset(dataset)` after loading the index. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import cagra
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = cagra.build(cagra.IndexParams(), dataset)
>>> # Serialize and deserialize the cagra index built
>>> cagra.save("my_index.bin", index)
>>> index_loaded = cagra.load("my_index.bin")
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
| `search_params` | `SearchParams` |  |
| `index` | `Index` | Trained CAGRA index. |
| `queries` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float, int8, uint8] |
| `k` | `int` | The number of neighbors. |
| `neighbors` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k), dtype int64_t. If supplied, neighbor indices will be written here in-place. (default None) |
| `distances` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k) If supplied, the distances to the neighbors will be written here in-place. (default None) |
| `filter` | `Optional cuvs.neighbors.cuvsFilter can be used to filter` | neighbors based on a given bitset. (default None) |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import cagra
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = cagra.build(cagra.IndexParams(), dataset)
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> k = 10
>>> search_params = cagra.SearchParams(
...     max_queries=100,
...     itopk_size=64
... )
>>> # Using a pooling allocator reduces overhead of temporary array
>>> # creation during search. This is useful if multiple searches
>>> # are performed with same query size.
>>> distances, neighbors = cagra.search(search_params, index, queries,
...                                     k)
>>> neighbors = cp.asarray(neighbors)
>>> distances = cp.asarray(distances)
```
