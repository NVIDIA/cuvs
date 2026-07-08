---
slug: api-reference/c-api-neighbors-hnsw
---

# HNSW

_Source header: `cuvs/neighbors/hnsw.h`_

## C API for HNSW index params

<a id="cuvshnswhierarchy"></a>
### cuvsHnswHierarchy

Hierarchy for HNSW index when converting from CAGRA index

NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only index. Such an index is immutable (it cannot be extended) and can only be read by the hnswlib wrapper in cuVS, because its serialization format is not compatible with the original hnswlib. Indexes built with hierarchy `CPU` or `GPU` are mutable and compatible with the original hnswlib library.

```c
enum cuvsHnswHierarchy {
  NONE = 0,
  CPU = 1,
  GPU = 2
};
```

**Values**

| Name | Value |
| --- | --- |
| `NONE` | `0` |
| `CPU` | `1` |
| `GPU` | `2` |

<a id="cuvshnswaceparams"></a>
### cuvsHnswAceParams

Parameters for ACE (Augmented Core Extraction) graph build for HNSW.

ACE enables building indexes for datasets too large to fit in GPU memory by:

1. Partitioning the dataset in core and augmented partitions using balanced k-means
2. Building sub-indexes for each partition independently
3. Concatenating sub-graphs into a final unified index

```c
struct cuvsHnswAceParams {
  size_t npartitions;
  const char* build_dir;
  bool use_disk;
  double max_host_memory_gb;
  double max_gpu_memory_gb;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `npartitions` | `size_t` | Number of partitions for ACE partitioned build.<br /><br />When set to 0 (default), the number of partitions is automatically derived based on available host and GPU memory to maximize partition size while ensuring the build fits in memory.<br /><br />Small values might improve recall but potentially degrade performance and increase memory usage. The partition size is on average 2 * (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests).<br /><br />If the specified number of partitions results in partitions that exceed available memory, the value will be automatically increased to fit memory constraints and a warning will be issued. |
| `build_dir` | `const char*` | Directory to store ACE build artifacts (e.g., KNN graph, optimized graph). Used when `use_disk` is true or when the graph does not fit in memory. |
| `use_disk` | `bool` | Whether to use disk-based storage for ACE build. When true, enables disk-based operations for memory-efficient graph construction. |
| `max_host_memory_gb` | `double` | Maximum host memory to use for ACE build in GiB. When set to 0 (default), uses available host memory. Useful for testing or when running alongside other memory-intensive processes. |
| `max_gpu_memory_gb` | `double` | Maximum GPU memory to use for ACE build in GiB. When set to 0 (default), uses available GPU memory. Useful for testing or when running alongside other memory-intensive processes. |

<a id="cuvshnswaceparamscreate"></a>
### cuvsHnswAceParamsCreate

Allocate HNSW ACE params, and populate with default values

```c
cuvsError_t cuvsHnswAceParamsCreate(cuvsHnswAceParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsHnswAceParams_t*`](/api-reference/c-api-neighbors-hnsw#cuvshnswaceparams) | cuvsHnswAceParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvshnswaceparamsdestroy"></a>
### cuvsHnswAceParamsDestroy

De-allocate HNSW ACE params

```c
cuvsError_t cuvsHnswAceParamsDestroy(cuvsHnswAceParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsHnswAceParams_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswaceparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvshnswindexparamscreate"></a>
### cuvsHnswIndexParamsCreate

Allocate HNSW Index params, and populate with default values

```c
cuvsError_t cuvsHnswIndexParamsCreate(cuvsHnswIndexParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswIndexParams_t*` | cuvsHnswIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvshnswindexparamsdestroy"></a>
### cuvsHnswIndexParamsDestroy

De-allocate HNSW Index params

```c
cuvsError_t cuvsHnswIndexParamsDestroy(cuvsHnswIndexParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswIndexParams_t` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## C API for hnswlib wrapper index

<a id="cuvshnswindex"></a>
### cuvsHnswIndex

Struct to hold address of cuvs::neighbors::Hnsw::index and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsHnswIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvshnswindexcreate"></a>
### cuvsHnswIndexCreate

Allocate HNSW index

```c
cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsHnswIndex_t*`](/api-reference/c-api-neighbors-hnsw#cuvshnswindex) | cuvsHnswIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvshnswindexdestroy"></a>
### cuvsHnswIndexDestroy

De-allocate HNSW index

```c
cuvsError_t cuvsHnswIndexDestroy(cuvsHnswIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsHnswIndex_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswindex) | cuvsHnswIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Parameters for extending HNSW index

<a id="cuvshnswextendparams"></a>
### cuvsHnswExtendParams

Parameters for extending HNSW index

```c
struct cuvsHnswExtendParams {
  int num_threads;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `num_threads` | `int` | Number of CPU threads used to extend additional vectors |

<a id="cuvshnswextendparamscreate"></a>
### cuvsHnswExtendParamsCreate

Allocate HNSW extend params, and populate with default values

```c
cuvsError_t cuvsHnswExtendParamsCreate(cuvsHnswExtendParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsHnswExtendParams_t*`](/api-reference/c-api-neighbors-hnsw#cuvshnswextendparams) | cuvsHnswExtendParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvshnswextendparamsdestroy"></a>
### cuvsHnswExtendParamsDestroy

De-allocate HNSW extend params

```c
cuvsError_t cuvsHnswExtendParamsDestroy(cuvsHnswExtendParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsHnswExtendParams_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswextendparams) | cuvsHnswExtendParams_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Load CAGRA index as hnswlib index

<a id="cuvshnswfromcagra"></a>
### cuvsHnswFromCagra

Convert a CAGRA Index to an HNSW index. NOTE: When hierarchy is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.
3. `GPU`: The returned index is mutable, and its hierarchy is built on the GPU. The serialized index is also compatible with the original hnswlib library.

```c
cuvsError_t cuvsHnswFromCagra(cuvsResources_t res,
cuvsHnswIndexParams_t params,
cuvsCagraIndex_t cagra_index,
cuvsHnswIndex_t hnsw_index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | `cuvsHnswIndexParams_t` | cuvsHnswIndexParams_t used to load Hnsw index |
| `cagra_index` | in | [`cuvsCagraIndex_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindex) | cuvsCagraIndex_t to convert to HNSW index |
| `hnsw_index` | out | [`cuvsHnswIndex_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswindex) | cuvsHnswIndex_t to return the HNSW index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Build HNSW index on the GPU

<a id="cuvshnswbuild"></a>
### cuvsHnswBuild

Build an HNSW index on the GPU and search it on the CPU.

```c
cuvsError_t cuvsHnswBuild(cuvsResources_t res,
cuvsHnswIndexParams_t params,
DLManagedTensor* dataset,
cuvsHnswIndex_t index);
```

cuVS accepts HNSW parameters (`M`, `ef_construction`, hierarchy, and metric), builds a compatible graph on the GPU, and returns an HNSW index for CPU search. Internally, cuVS may use CAGRA graph construction and convert the graph to HNSW format. Users can leave `params-&gt;ace_params` as nullptr for the default in-memory path, or set ACE parameters to request partitioned or disk-backed graph construction.

NOTE: This function requires CUDA to be available at runtime.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | `cuvsHnswIndexParams_t` | cuvsHnswIndexParams_t with HNSW build parameters |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* host dataset to build index from |
| `index` | out | [`cuvsHnswIndex_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswindex) | cuvsHnswIndex_t to return the built HNSW index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Extend HNSW index with additional vectors

<a id="cuvshnswextend"></a>
### cuvsHnswExtend

Add new vectors to an HNSW index NOTE: A base-layer-only index with hierarchy `NONE` cannot be extended. Indexes with hierarchy `CPU` or `GPU` can be extended.

```c
cuvsError_t cuvsHnswExtend(cuvsResources_t res,
cuvsHnswExtendParams_t params,
DLManagedTensor* additional_dataset,
cuvsHnswIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsHnswExtendParams_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswextendparams) | cuvsHnswExtendParams_t used to extend Hnsw index |
| `additional_dataset` | in | `DLManagedTensor*` | DLManagedTensor* additional dataset to extend the index |
| `index` | inout | [`cuvsHnswIndex_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswindex) | cuvsHnswIndex_t to extend |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## C API for hnswlib wrapper search params

<a id="cuvshnswsearchparams"></a>
### cuvsHnswSearchParams

C API for hnswlib wrapper search params

```c
struct cuvsHnswSearchParams {
  int32_t ef;
  int32_t num_threads;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `ef` | `int32_t` |  |
| `num_threads` | `int32_t` |  |

<a id="cuvshnswsearchparamscreate"></a>
### cuvsHnswSearchParamsCreate

Allocate HNSW search params, and populate with default values

```c
cuvsError_t cuvsHnswSearchParamsCreate(cuvsHnswSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsHnswSearchParams_t*`](/api-reference/c-api-neighbors-hnsw#cuvshnswsearchparams) | cuvsHnswSearchParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvshnswsearchparamsdestroy"></a>
### cuvsHnswSearchParamsDestroy

De-allocate HNSW search params

```c
cuvsError_t cuvsHnswSearchParamsDestroy(cuvsHnswSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsHnswSearchParams_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswsearchparams) | cuvsHnswSearchParams_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## C API for CUDA ANN Graph-based nearest neighbor search

<a id="cuvshnswsearch"></a>
### cuvsHnswSearch

Search a HNSW index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCPU`, `kDLCUDAHost`, or `kDLCUDAManaged`. It is also important to note that the HNSW Index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` and `index.dtype.bits == queries.dl_tensor.dtype.bits` Supported types for input are:

1. `queries`: a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` b. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16` c. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8` d. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 64`
3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` NOTE: When hierarchy is `NONE`, the HNSW index can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.

```c
cuvsError_t cuvsHnswSearch(cuvsResources_t res,
cuvsHnswSearchParams_t params,
cuvsHnswIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsHnswSearchParams_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswsearchparams) | cuvsHnswSearchParams_t used to search Hnsw index |
| `index` | in | [`cuvsHnswIndex_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswindex) | cuvsHnswIndex returned by `cuvsHnswBuild`, `cuvsHnswFromCagra`, or `cuvsHnswDeserialize` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## HNSW C-API serialize functions

<a id="cuvshnswserialize"></a>
### cuvsHnswSerialize

Serialize an HNSW index to a file. NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU` or `GPU`, the saved hnswlib index is compatible with the original hnswlib library.

```c
cuvsError_t cuvsHnswSerialize(cuvsResources_t res, const char* filename, cuvsHnswIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file to save the index |
| `index` | in | [`cuvsHnswIndex_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswindex) | cuvsHnswIndex_t to serialize |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvshnswdeserialize"></a>
### cuvsHnswDeserialize

Load hnswlib index from file which was serialized from a HNSW index.

```c
cuvsError_t cuvsHnswDeserialize(cuvsResources_t res,
cuvsHnswIndexParams_t params,
const char* filename,
int dim,
cuvsDistanceType metric,
cuvsHnswIndex_t index);
```

NOTE: When hierarchy is `NONE`, the loaded hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. Indexes loaded with hierarchy `CPU` or `GPU` are mutable and compatible with the original hnswlib library. Experimental, both the API and the serialization format are subject to change. NOTE: `params-&gt;hierarchy` must match the hierarchy the index was serialized with. Loading a base-layer-only index saved with hierarchy `NONE` requires setting `params-&gt;hierarchy = NONE` (the default created by `cuvsHnswIndexParamsCreate` is `GPU`).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | `cuvsHnswIndexParams_t` | cuvsHnswIndexParams_t used to load Hnsw index; `params-&gt;hierarchy` must match the hierarchy the index was serialized with |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `dim` | in | `int` | the dimension of the vectors in the index |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | the distance metric used to build the index; takes precedence over `params-&gt;metric` |
| `index` | out | [`cuvsHnswIndex_t`](/api-reference/c-api-neighbors-hnsw#cuvshnswindex) | HNSW index loaded disk |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
