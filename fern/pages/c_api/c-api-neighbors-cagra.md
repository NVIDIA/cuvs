---
slug: api-reference/c-api-neighbors-cagra
---

# Cagra

_Source header: `cuvs/neighbors/cagra.h`_

## C API for CUDA ANN Graph-based nearest neighbor search

<a id="cuvscagragraphbuildalgo"></a>
### cuvsCagraGraphBuildAlgo

Enum to denote which ANN algorithm is used to build CAGRA graph

```c
enum cuvsCagraGraphBuildAlgo {
  AUTO_SELECT = 0,
  IVF_PQ = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `AUTO_SELECT` | `0` |
| `IVF_PQ` | `1` |

<a id="cuvscagrahnswheuristictype"></a>
### cuvsCagraHnswHeuristicType

A strategy for selecting the graph build parameters based on similar HNSW index parameters.

Define how cuvsCagraIndexParamsFromHnswParams should construct a graph to construct a graph that is to be converted to (used by) a CPU HNSW index.

```c
enum cuvsCagraHnswHeuristicType;
```

<a id="cuvscagracompressionparams"></a>
### cuvsCagraCompressionParams

Parameters for VPQ compression.

```c
struct cuvsCagraCompressionParams {
  uint32_t pq_bits;
  uint32_t pq_dim;
  uint32_t vq_n_centers;
  uint32_t kmeans_n_iters;
  double vq_kmeans_trainset_fraction;
  double pq_kmeans_trainset_fraction;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ.<br /><br />Possible values: [4, 5, 6, 7, 8].<br /><br />Hint: the smaller the 'pq_bits', the smaller the index size and the better the search performance, but the lower the recall. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is selected using a heuristic.<br /><br />TODO: at the moment `dim` must be a multiple `pq_dim`. |
| `vq_n_centers` | `uint32_t` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". When zero, an optimal value is selected using a heuristic. |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (both VQ & PQ phases). |
| `vq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (VQ phase). When zero, an optimal value is selected using a heuristic. |
| `pq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (PQ phase). When zero, an optimal value is selected using a heuristic. |

<a id="cuvsaceparams"></a>
### cuvsAceParams

Parameters for ACE (Augmented Core Extraction) graph build.

ACE enables building indexes for datasets too large to fit in GPU memory by:

1. Partitioning the dataset in core (closest) and augmented (second-closest) partitions using balanced k-means.
2. Building sub-indexes for each partition independently
3. Concatenating sub-graphs into a final unified index

```c
struct cuvsAceParams {
  size_t npartitions;
  size_t ef_construction;
  const char* build_dir;
  bool use_disk;
  double max_host_memory_gb;
  double max_gpu_memory_gb;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `npartitions` | `size_t` | Number of partitions for ACE (Augmented Core Extraction) partitioned build.<br /><br />When set to 0 (default), the number of partitions is automatically derived based on available host and GPU memory to maximize partition size while ensuring the build fits in memory.<br /><br />Small values might improve recall but potentially degrade performance and increase memory usage. Partitions should not be too small to prevent issues in KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests).<br /><br />If the specified number of partitions results in partitions that exceed available memory, the value will be automatically increased to fit memory constraints and a warning will be issued. |
| `ef_construction` | `size_t` | The index quality for the ACE build.<br /><br />Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| `build_dir` | `const char*` | Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).<br /><br />Used when `use_disk` is true or when the graph does not fit in host and GPU memory. This should be the fastest disk in the system and hold enough space for twice the dataset, final graph, and label mapping. The directory may already exist, but ACE's named artifacts must not already exist. Simultaneous builds must use different directories. On failure, ACE removes only artifacts it created and never deletes unrelated directory contents. |
| `use_disk` | `bool` | Whether to use disk-based storage for ACE build.<br /><br />When true, enables disk-based operations for memory-efficient graph construction. |
| `max_host_memory_gb` | `double` | Maximum host memory to use for ACE build in GiB.<br /><br />When set to 0 (default), uses available host memory. When set to a positive value, limits host memory usage to the specified amount. Useful for testing or when running alongside other memory-intensive processes. |
| `max_gpu_memory_gb` | `double` | Maximum GPU memory to use for ACE build in GiB.<br /><br />When set to 0 (default), uses available GPU memory. When set to a positive value, limits GPU memory usage to the specified amount. Useful for testing or when running alongside other memory-intensive processes. |

<a id="cuvscagraindexparams"></a>
### cuvsCagraIndexParams

Supplemental parameters to build CAGRA Index

```c
struct cuvsCagraIndexParams {
  cuvsDistanceType metric;
  size_t intermediate_graph_degree;
  size_t graph_degree;
  enum cuvsCagraGraphBuildAlgo build_algo;
  size_t nn_descent_niter;
  void* graph_build_params;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | Distance type. |
| `intermediate_graph_degree` | `size_t` | Degree of input graph for pruning. |
| `graph_degree` | `size_t` | Degree of output graph. |
| `build_algo` | [`enum cuvsCagraGraphBuildAlgo`](/api-reference/c-api-neighbors-cagra#cuvscagragraphbuildalgo) | ANN algorithm to build knn graph. |
| `nn_descent_niter` | `size_t` | Number of Iterations to run if building with NN_DESCENT |
| `graph_build_params` | `void*` | Optional: specify graph build params based on build_algo<br />- IVF_PQ: cuvsIvfPqParams_t<br />- ACE: cuvsAceParams_t<br />- Others: nullptr |

<a id="cuvscagramergealgo"></a>
### cuvsCagraMergeAlgo

Algorithm used to merge physical CAGRA indices.

```c
enum cuvsCagraMergeAlgo {
  CUVS_CAGRA_MERGE_AUTO = 0,
  CUVS_CAGRA_MERGE_FASTENER = 1,
  CUVS_CAGRA_MERGE_REBUILD = 2
};
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_CAGRA_MERGE_AUTO` | `0` |
| `CUVS_CAGRA_MERGE_FASTENER` | `1` |
| `CUVS_CAGRA_MERGE_REBUILD` | `2` |

<a id="cuvscagramergeparams"></a>
### cuvsCagraMergeParams

Parameters controlling how physical CAGRA indices are merged.

```c
struct cuvsCagraMergeParams {
  uint32_t levels;
  uint32_t root_fanout;
  uint32_t lower_fanout;
  double leader_fraction;
  uint32_t max_leaders;
  uint32_t leaf_size;
  uint32_t leaf_degree;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `levels` | `uint32_t` |  |
| `root_fanout` | `uint32_t` |  |
| `lower_fanout` | `uint32_t` |  |
| `leader_fraction` | `double` |  |
| `max_leaders` | `uint32_t` |  |
| `leaf_size` | `uint32_t` |  |
| `leaf_degree` | `uint32_t` |  |

<a id="cuvscagraindexparamscreate"></a>
### cuvsCagraIndexParamsCreate

Allocate CAGRA Index params, and populate with default values

```c
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsCagraIndexParams_t*`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) | cuvsCagraIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexparamsdestroy"></a>
### cuvsCagraIndexParamsDestroy

De-allocate CAGRA Index params

```c
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsCagraIndexParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagramergeparamscreate"></a>
### cuvsCagraMergeParamsCreate

Allocate CAGRA merge params and populate them with AUTO defaults.

```c
cuvsError_t cuvsCagraMergeParamsCreate(cuvsCagraMergeParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` |  | [`cuvsCagraMergeParams_t*`](/api-reference/c-api-neighbors-cagra#cuvscagramergeparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagramergeparamsdestroy"></a>
### cuvsCagraMergeParamsDestroy

De-allocate CAGRA merge params.

```c
cuvsError_t cuvsCagraMergeParamsDestroy(cuvsCagraMergeParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` |  | [`cuvsCagraMergeParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagramergeparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagracompressionparamscreate"></a>
### cuvsCagraCompressionParamsCreate

Allocate CAGRA Compression params, and populate with default values

```c
cuvsError_t cuvsCagraCompressionParamsCreate(cuvsCagraCompressionParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsCagraCompressionParams_t*`](/api-reference/c-api-neighbors-cagra#cuvscagracompressionparams) | cuvsCagraCompressionParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagracompressionparamsdestroy"></a>
### cuvsCagraCompressionParamsDestroy

De-allocate CAGRA Compression params

```c
cuvsError_t cuvsCagraCompressionParamsDestroy(cuvsCagraCompressionParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsCagraCompressionParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagracompressionparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsaceparamscreate"></a>
### cuvsAceParamsCreate

Allocate ACE params, and populate with default values

```c
cuvsError_t cuvsAceParamsCreate(cuvsAceParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsAceParams_t*`](/api-reference/c-api-neighbors-cagra#cuvsaceparams) | cuvsAceParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsaceparamsdestroy"></a>
### cuvsAceParamsDestroy

De-allocate ACE params

```c
cuvsError_t cuvsAceParamsDestroy(cuvsAceParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsAceParams_t`](/api-reference/c-api-neighbors-cagra#cuvsaceparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexparamsfromhnswparams"></a>
### cuvsCagraIndexParamsFromHnswParams

Create CAGRA index parameters similar to an HNSW index

```c
cuvsError_t cuvsCagraIndexParamsFromHnswParams(cuvsCagraIndexParams_t params,
int64_t n_rows,
int64_t dim,
int M,
int ef_construction,
enum cuvsCagraHnswHeuristicType heuristic,
cuvsDistanceType metric);
```

This factory function creates CAGRA parameters that yield a graph compatible with an HNSW graph with the given parameters.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | out | [`cuvsCagraIndexParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) | The CAGRA index params to populate |
| `n_rows` | in | `int64_t` | Number of rows in the dataset |
| `dim` | in | `int64_t` | Number of dimensions in the dataset |
| `M` | in | `int` | HNSW index parameter M |
| `ef_construction` | in | `int` | HNSW index parameter ef_construction |
| `heuristic` | in | [`enum cuvsCagraHnswHeuristicType`](/api-reference/c-api-neighbors-cagra#cuvscagrahnswheuristictype) | Strategy for parameter selection |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | Distance metric to use |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexparamsfromdataset"></a>
### cuvsCagraIndexParamsFromDataset

Create CAGRA index parameters heuristically tuned for a dataset

```c
cuvsError_t cuvsCagraIndexParamsFromDataset(cuvsCagraIndexParams_t params,
int64_t n_rows,
int64_t dim,
size_t graph_degree,
cuvsDistanceType metric,
size_t build_quality);
```

This factory function selects the graph build algorithm and its parameters based on the shape of the dataset.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | out | [`cuvsCagraIndexParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) | The CAGRA index params to populate |
| `n_rows` | in | `int64_t` | Number of rows in the dataset |
| `dim` | in | `int64_t` | Number of dimensions in the dataset |
| `graph_degree` | in | `size_t` | Degree of the output graph |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | Distance metric to use |
| `build_quality` | in | `size_t` | Higher values increase build quality (and cost) up to a point |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## C API for CUDA ANN Graph-based nearest neighbor search

<a id="cuvscagraextendparams"></a>
### cuvsCagraExtendParams

Supplemental parameters to extend CAGRA Index

```c
struct cuvsCagraExtendParams {
  uint32_t max_chunk_size;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `max_chunk_size` | `uint32_t` | The additional dataset is divided into chunks and added to the graph. This is the knob to adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also degrade recall because no edges are added between the nodes in the same chunk. Auto select when 0. |

<a id="cuvscagraextendparamscreate"></a>
### cuvsCagraExtendParamsCreate

Allocate CAGRA Extend params, and populate with default values

```c
cuvsError_t cuvsCagraExtendParamsCreate(cuvsCagraExtendParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsCagraExtendParams_t*`](/api-reference/c-api-neighbors-cagra#cuvscagraextendparams) | cuvsCagraExtendParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraextendparamsdestroy"></a>
### cuvsCagraExtendParamsDestroy

De-allocate CAGRA Extend params

```c
cuvsError_t cuvsCagraExtendParamsDestroy(cuvsCagraExtendParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsCagraExtendParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraextendparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraextend"></a>
### cuvsCagraExtend

Extend a CAGRA index with a caller-owned pre-concatenated padded dataset.

```c
cuvsError_t cuvsCagraExtend(cuvsResources_t res,
cuvsCagraExtendParams_t params,
cuvsDataset_t extended_dataset,
int64_t new_start_row,
cuvsCagraIndex_t index);
```

The caller must build `extended_dataset` as `old \|\| new` (size `n_old + n_new`) before calling. Rows `[0, new_start_row)` are the original vectors; rows `[new_start_row, n_rows)` are the additional vectors. `new_start_row` must equal the current index size. The library only extends the graph and rebinds the index to `extended_dataset`; keep that dataset alive for the index lifetime.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsCagraExtendParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraextendparams) | cuvsCagraExtendParams_t used to extend CAGRA index |
| `extended_dataset` | in | `cuvsDataset_t` | cuvsDataset_t caller-owned device-padded dataset of old \|\| new |
| `new_start_row` | in | `int64_t` | row index where the additional vectors begin |
| `index` | in,out | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | cuvsCagraIndex_t CAGRA index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## C API for CUDA ANN Graph-based nearest neighbor search

<a id="cuvscagrasearchalgo"></a>
### cuvsCagraSearchAlgo

Enum to denote algorithm used to search CAGRA Index

```c
enum cuvsCagraSearchAlgo {
  SINGLE_CTA = 0,
  MULTI_CTA = 1,
  MULTI_KERNEL = 2,
  AUTO = 100
};
```

**Values**

| Name | Value |
| --- | --- |
| `SINGLE_CTA` | `0` |
| `MULTI_CTA` | `1` |
| `MULTI_KERNEL` | `2` |
| `AUTO` | `100` |

<a id="cuvscagrahashmode"></a>
### cuvsCagraHashMode

Enum to denote Hash Mode used while searching CAGRA index

```c
enum cuvsCagraHashMode {
  HASH = 0,
  SMALL = 1,
  AUTO_HASH = 100
};
```

**Values**

| Name | Value |
| --- | --- |
| `HASH` | `0` |
| `SMALL` | `1` |
| `AUTO_HASH` | `100` |

<a id="cuvscagrasearchparams"></a>
### cuvsCagraSearchParams

Supplemental parameters to search CAGRA index

```c
struct cuvsCagraSearchParams {
  size_t max_queries;
  size_t itopk_size;
  size_t max_iterations;
  enum cuvsCagraSearchAlgo algo;
  size_t team_size;
  size_t search_width;
  size_t min_iterations;
  size_t thread_block_size;
  enum cuvsCagraHashMode hashmap_mode;
  size_t hashmap_min_bitlen;
  float hashmap_max_fill_rate;
  uint32_t num_random_samplings;
  uint64_t rand_xor_mask;
  bool persistent;
  float persistent_lifetime;
  float persistent_device_usage;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `max_queries` | `size_t` | Maximum number of queries to search at the same time (batch size). Auto select when 0. |
| `itopk_size` | `size_t` | Number of intermediate search results retained during the search.<br /><br />This is the main knob to adjust trade off between accuracy and search speed. Higher values improve the search accuracy. |
| `max_iterations` | `size_t` | Upper limit of search iterations. Auto select when 0. |
| `algo` | [`enum cuvsCagraSearchAlgo`](/api-reference/c-api-neighbors-cagra#cuvscagrasearchalgo) | Which search implementation to use. |
| `team_size` | `size_t` | Number of threads used to calculate a single distance. 4, 8, 16, or 32. |
| `search_width` | `size_t` | Number of graph nodes to select as the starting point for the search in each iteration. aka search width? |
| `min_iterations` | `size_t` | Lower limit of search iterations. |
| `thread_block_size` | `size_t` | Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. |
| `hashmap_mode` | [`enum cuvsCagraHashMode`](/api-reference/c-api-neighbors-cagra#cuvscagrahashmode) | Hashmap type. Auto selection when AUTO. |
| `hashmap_min_bitlen` | `size_t` | Lower limit of hashmap bit length. More than 8. |
| `hashmap_max_fill_rate` | `float` | Upper limit of hashmap fill rate. More than 0.1, less than 0.9. |
| `num_random_samplings` | `uint32_t` | Number of iterations of initial random seed node selection. 1 or more. |
| `rand_xor_mask` | `uint64_t` | Bit mask used for initial random seed node selection. |
| `persistent` | `bool` | Whether to use the persistent version of the kernel (only SINGLE_CTA is supported a.t.m.) |
| `persistent_lifetime` | `float` | Persistent kernel: time in seconds before the kernel stops if no requests received. |
| `persistent_device_usage` | `float` | Set the fraction of maximum grid size used by persistent kernel. Value 1.0 means the kernel grid size is maximum possible for the selected device. The value must be greater than 0.0 and not greater than 1.0.<br /><br />One may need to run other kernels alongside this persistent kernel. This parameter can be used to reduce the grid size of the persistent kernel to leave a few SMs idle.<br />Note: running any other work on GPU alongside with the persistent kernel makes the setup fragile.<br />- Running another kernel in another thread usually works, but no progress guaranteed<br />- Any CUDA allocations block the context (this issue may be obscured by using pools)<br />- Memory copies to not-pinned host memory may block the context<br /><br />Even when we know there are no other kernels working at the same time, setting kDeviceUsage to 1.0 surprisingly sometimes hurts performance. Proceed with care. If you suspect this is an issue, you can reduce this number to ~0.9 without a significant impact on the throughput. |

<a id="cuvscagrasearchparamscreate"></a>
### cuvsCagraSearchParamsCreate

Allocate CAGRA search params, and populate with default values

```c
cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsCagraSearchParams_t*`](/api-reference/c-api-neighbors-cagra#cuvscagrasearchparams) | cuvsCagraSearchParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagrasearchparamsdestroy"></a>
### cuvsCagraSearchParamsDestroy

De-allocate CAGRA search params

```c
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsCagraSearchParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagrasearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## C API for CUDA ANN Graph-based nearest neighbor search

<a id="cuvscagraindex"></a>
### cuvsCagraIndex

Struct holding the CAGRA index storage address and vector element dtype (DLPack-style)

Matches the usual cuVS C index pattern (`addr` + `dtype`). `addr` points at implementation-owned storage (not always a bare `cagra::index*`); free only via `cuvsCagraIndexDestroy`. `dtype` describes index vector elements for queries and template dispatch.

```c
typedef struct cuvsCagraIndex {
  uintptr_t addr;
  DLDataType dtype;
} cuvsCagraIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvscagraindexcreate"></a>
### cuvsCagraIndexCreate

Allocate CAGRA index

```c
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsCagraIndex_t*`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | cuvsCagraIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexdestroy"></a>
### cuvsCagraIndexDestroy

De-allocate CAGRA index

```c
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | cuvsCagraIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexgetdims"></a>
### cuvsCagraIndexGetDims

Get dimension of the CAGRA index

```c
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index |
| `dim` | out | `int64_t*` | return dimension of the index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexgetsize"></a>
### cuvsCagraIndexGetSize

Get size of the CAGRA index

```c
cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t* size);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index |
| `size` | out | `int64_t*` | return number of vectors in the index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexgetgraphdegree"></a>
### cuvsCagraIndexGetGraphDegree

Get graph degree of the CAGRA index

```c
cuvsError_t cuvsCagraIndexGetGraphDegree(cuvsCagraIndex_t index, int64_t* graph_degree);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index |
| `graph_degree` | out | `int64_t*` | return graph degree |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexgetdataset"></a>
### cuvsCagraIndexGetDataset

Returns a view of the CAGRA dataset

```c
cuvsError_t cuvsCagraIndexGetDataset(cuvsCagraIndex_t index, DLManagedTensor* dataset);
```

This function returns a non-owning view of the CAGRA dataset. The output will be referencing device memory that is directly used in CAGRA, without copying the dataset at all. This means that the output is only valid as long as the CAGRA index is alive, and once cuvsCagraIndexDestroy is called on the cagra index - the returned dataset view will be invalid.

Note that the DLManagedTensor dataset returned will have an associated 'deleter' function that must be called when the dataset is no longer needed. This will free up host memory that stores the shape of the dataset view.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index |
| `dataset` | out | `DLManagedTensor*` | the dataset used in cagra |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexgetgraph"></a>
### cuvsCagraIndexGetGraph

Returns a view of the CAGRA graph

```c
cuvsError_t cuvsCagraIndexGetGraph(cuvsCagraIndex_t index, DLManagedTensor* graph);
```

This function returns a non-owning view of the CAGRA graph. The output will be referencing device memory that is directly used in CAGRA, without copying the graph at all. This means that the output is only valid as long as the CAGRA index is alive, and once cuvsCagraIndexDestroy is called on the cagra index - the returned graph view will be invalid.

Note that the DLManagedTensor graph returned will have an associated 'deleter' function that must be called when the graph is no longer needed. This will free up host memory that stores the metadata for the graph view.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index |
| `graph` | out | `DLManagedTensor*` | the output knn graph. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraupdatedataset"></a>
### cuvsCagraUpdateDataset

Update a CAGRA index with a device-padded dataset.

```c
cuvsError_t cuvsCagraUpdateDataset(cuvsResources_t res,
cuvsDataset_t device_padded_dataset,
cuvsCagraIndex_t index);
```

This is the centralized dataset update operation for C callers. If `index` is already device-padded, its dataset view is replaced in place. Otherwise, the index is converted and its opaque handle is rebound to a search-ready device-padded index. Caller retains ownership of

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `device_padded_dataset` | in | `cuvsDataset_t` | owning or non-owning device-padded dataset handle |
| `index` | inout | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index handle |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## C API for CUDA ANN Graph-based nearest neighbor search

<a id="cuvscagrabuild"></a>
### cuvsCagraBuild

Build a CAGRA index from a dataset handle. Acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`

```c
cuvsError_t cuvsCagraBuild(cuvsResources_t res,
cuvsCagraIndexParams_t params,
cuvsDataset_t dataset,
cuvsCagraIndex_t index);
```

The memory space and layout `dataset` was constructed with select the C++ build overload. Build the handle with an owning factory or the matching dataset view factory (`cuvsDatasetMakePaddedView` / `cuvsDatasetMakeStandardView`).

Note that a dataset residing in host memory produces a host-backed index, which must be made search-ready with `cuvsCagraUpdateDataset` (using a device-padded dataset) before calling `cuvsCagraSearch`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsCagraIndexParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) | cuvsCagraIndexParams_t used to build CAGRA index |
| `dataset` | in | `cuvsDataset_t` | cuvsDataset_t training dataset or dataset view |
| `index` | inout | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | cuvsCagraIndex_t Newly built CAGRA index. This index needs to be already created with cuvsCagraIndexCreate. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## C API for CUDA ANN Graph-based nearest neighbor search

<a id="cuvscagrasearch"></a>
### cuvsCagraSearch

Search a CAGRA index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. It is also important to note that the CAGRA Index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Types for input are:

1. `queries`: a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` b. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16` c. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8` d. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32` or `kDLDataType.code == kDLInt`  and `kDLDataType.bits = 64`
3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

```c
cuvsError_t cuvsCagraSearch(cuvsResources_t res,
cuvsCagraSearchParams_t params,
cuvsCagraIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter filter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsCagraSearchParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagrasearchparams) | cuvsCagraSearchParams_t used to search CAGRA index |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | cuvsCagraIndex which has been returned by `cuvsCagraBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `filter` | in | [`cuvsFilter`](/api-reference/c-api-neighbors-common#cuvsfilter) | cuvsFilter input filter that can be used to filter queries and neighbors based on the given bitset. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagrasearchmultipartition"></a>
### cuvsCagraSearchMultiPartition

Search multiple CAGRA index partitions concurrently and return the global top-k per query.

```c
cuvsError_t cuvsCagraSearchMultiPartition(cuvsResources_t res,
cuvsCagraSearchParams_t params,
uint32_t num_partitions,
cuvsCagraIndex_t* indices,
DLManagedTensor* queries,
DLManagedTensor* partition_ids,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter* filters);
```

For each query row, the function searches all partitions in parallel into an internal intermediate buffer, applies per-partition distance post-processing, runs a batched top-k merge across partitions, and writes the final outputs to the caller-supplied device tensors. All work is submitted to the CUDA stream associated with `res`; use `cuvsStreamSync` to wait for completion.

The index element type may be float32, float16, int8, or uint8. All partitions must share the same element type, and the queries must use that same type.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsCagraSearchParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagrasearchparams) | search parameters (shared across partitions) |
| `num_partitions` | in | `uint32_t` | number of index partitions |
| `indices` | in | [`cuvsCagraIndex_t*`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | array of num_partitions cuvsCagraIndex_t pointers, all of the same element type |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* (device, same dtype as the indices, [n_queries, dim]); the queries matrix is searched against every partition |
| `partition_ids` | out | `DLManagedTensor*` | DLManagedTensor* (device, uint32, [n_queries, k]); which partition each returned neighbor came from |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* (device, uint32 or int64, [n_queries, k]); ordinal in the corresponding partition's dataset |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* (device, float32, [n_queries, k]); post-processed distance for each (query, neighbor) |
| `filters` | in | [`cuvsFilter*`](/api-reference/c-api-neighbors-common#cuvsfilter) | array of `num_partitions` filters, one per partition (or NULL for a fully unfiltered search). `filters[i]` applies to partition `i`: use \{.type=NO_FILTER, .addr=0\} for no filter on that partition, or \{.type=BITSET, .addr=ptr\} where ptr is a uintptr_t-cast DLManagedTensor* holding that partition's own bitset (one bit per vector in that partition; standard 32-bit packing). |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## CAGRA C-API serialize functions

<a id="cuvscagraserializegraph"></a>
### cuvsCagraSerializeGraph

Save the CAGRA graph to file without its dataset.

```c
cuvsError_t cuvsCagraSerializeGraph(cuvsResources_t res,
const char* filename,
cuvsCagraIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the graph |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraserializegraphanddataset"></a>
### cuvsCagraSerializeGraphAndDataset

Save the CAGRA graph and its attached dataset to file.

```c
cuvsError_t cuvsCagraSerializeGraphAndDataset(cuvsResources_t res,
const char* filename,
cuvsCagraIndex_t index);
```

The index stores a non-owning dataset view. The caller must keep the dataset backing that view alive while this function runs. Returns CUVS_ERROR without modifying the destination file if the index has no attached dataset.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the graph and dataset |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index with an attached host or device dataset |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraserializetohnswlib"></a>
### cuvsCagraSerializeToHnswlib

Save the CAGRA index to file in hnswlib format.

```c
cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
const char* filename,
cuvsCagraIndex_t index);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | CAGRA index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagradeserializegraph"></a>
### cuvsCagraDeserializeGraph

Load the CAGRA graph from file without retaining a serialized dataset.

```c
cuvsError_t cuvsCagraDeserializeGraph(cuvsResources_t res,
const char* filename,
cuvsCagraIndex_t index);
```

This succeeds whether or not the file contains a dataset. Use cuvsCagraUpdateDataset to attach a caller-owned device-padded dataset view before searching the graph-only index.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | inout | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | pre-created CAGRA index populated on success and unchanged on failure |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagradeserializegraphanddataset"></a>
### cuvsCagraDeserializeGraphAndDataset

Load the CAGRA graph and dataset from file.

```c
cuvsError_t cuvsCagraDeserializeGraphAndDataset(cuvsResources_t res,
const char* filename,
cuvsCagraIndex_t index,
cuvsDataset_t* out_dataset);
```

The returned owning dataset preserves the serialized host/device memory type and standard/padded layout. The index stores a non-owning view into it, so the caller must keep the dataset alive while the index uses it and destroy it separately with cuvsDatasetDestroy. Only a device-padded result is immediately searchable through the C API; attach a caller-owned device-padded view with cuvsCagraUpdateDataset for any other kind. The output pointer must point to a null handle on entry; deserialization acts as a factory and transfers ownership of the allocated dataset handle on success. Returns CUVS_ERROR when the file has no dataset; the index and output handle are unchanged on failure.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the graph and dataset |
| `index` | inout | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | pre-created CAGRA index populated on success and unchanged on failure |
| `out_dataset` | out | `cuvsDataset_t*` | receives the allocated owning dataset handle; must point to null on entry |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagraindexfromargs"></a>
### cuvsCagraIndexFromArgs

Load index from a dataset and graph

```c
cuvsError_t cuvsCagraIndexFromArgs(cuvsResources_t res,
cuvsDistanceType metric,
DLManagedTensor* graph,
DLManagedTensor* dataset,
cuvsCagraIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | cuvsDistanceType to use in the index |
| `graph` | in | `DLManagedTensor*` | the knn graph to use, shape (size, graph_degree) |
| `dataset` | in | `DLManagedTensor*` | the dataset to use, shape (size, dim) |
| `index` | inout | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | cuvsCagraIndex_t CAGRA index populated with the graph and dataset. This index needs to be already created with cuvsCagraIndexCreate. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## CAGRA C-API merge functions

<a id="cuvscagramerge"></a>
### cuvsCagraMerge

Merge multiple CAGRA indices into a single CAGRA index.

```c
cuvsError_t cuvsCagraMerge(cuvsResources_t res,
cuvsCagraIndexParams_t params,
cuvsCagraIndex_t* indices,
size_t num_indices,
cuvsFilter filter,
cuvsDataset_t merged_dataset,
cuvsCagraIndex_t output_index);
```

All input indices must have been built with the same data type (`index.dtype`) and have the same dimensionality (`index.dims`). The merged index uses the output parameters specified in `cuvsCagraIndexParams`. The merge algorithm is selected automatically.

Input indices must have:

- `index.dtype.code` and `index.dtype.bits` matching across all indices.
- Supported data types for indices: a. `kDLFloat` with `bits = 32` b. `kDLFloat` with `bits = 16` c. `kDLInt` with `bits = 8` d. `kDLUInt` with `bits = 8`

The resulting output index will have the same data type as the input indices.

Example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsCagraIndexParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) | cuvsCagraIndexParams_t parameters for the output index |
| `indices` | in | [`cuvsCagraIndex_t*`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | Array of input cuvsCagraIndex_t handles to merge |
| `num_indices` | in | `size_t` | Number of input indices |
| `filter` | in | [`cuvsFilter`](/api-reference/c-api-neighbors-common#cuvsfilter) | Filter that can be used to filter out vectors from the merged index |
| `merged_dataset` | out | `cuvsDataset_t` | Empty owning dataset handle. Merge first attempts to allocate and populate device storage with the same layout as the input indices. For an unfiltered merge, if device allocation fails, it falls back to host storage and returns a host-backed output index. Keep this dataset alive while using `output_index`. A host-backed output index must be updated with `cuvsCagraUpdateDataset` before device search. |
| `output_index` | out | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | Output handle that will store the merged index. Must be initialized using `cuvsCagraIndexCreate` before use. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscagramergewithparams"></a>
### cuvsCagraMergeWithParams

Merge multiple CAGRA indices with explicit merge parameters.

```c
cuvsError_t cuvsCagraMergeWithParams(cuvsResources_t res,
cuvsCagraIndexParams_t params,
cuvsCagraMergeParams_t merge_params,
cuvsCagraIndex_t* indices,
size_t num_indices,
cuvsFilter filter,
cuvsDataset_t merged_dataset,
cuvsCagraIndex_t output_index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsCagraIndexParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) | cuvsCagraIndexParams_t parameters for the output index |
| `merge_params` | in | [`cuvsCagraMergeParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagramergeparams) | cuvsCagraMergeParams_t parameters controlling the merge algorithm, or NULL to use AUTO defaults |
| `indices` | in | [`cuvsCagraIndex_t*`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | Array of input cuvsCagraIndex_t handles to merge |
| `num_indices` | in | `size_t` | Number of input indices |
| `filter` | in | [`cuvsFilter`](/api-reference/c-api-neighbors-common#cuvsfilter) | Filter that can be used to filter out vectors from the merged index |
| `merged_dataset` | out | `cuvsDataset_t` | Empty owning dataset handle. Merge first attempts to allocate and populate device storage with the same layout as the input indices. For an unfiltered merge, AUTO and REBUILD can fall back to host storage if device allocation fails; explicit FASTENER reports the allocation failure instead. Keep this dataset alive while using `output_index`. A host-backed output index must be updated with `cuvsCagraUpdateDataset` before device search. |
| `output_index` | out | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | Output handle initialized with `cuvsCagraIndexCreate` |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
