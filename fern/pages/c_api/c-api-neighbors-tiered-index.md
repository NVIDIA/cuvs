---
slug: api-reference/c-api-neighbors-tiered-index
---

# Tiered Index

_Source header: `cuvs/neighbors/tiered_index.h`_

## Types

<a id="cuvstieredindexannalgo"></a>
### cuvsTieredIndexANNAlgo

Enum to hold which ANN algorithm is being used in the tiered index

```c
typedef enum {
  CUVS_TIERED_INDEX_ALGO_CAGRA = 0,
  CUVS_TIERED_INDEX_ALGO_IVF_FLAT = 1,
  CUVS_TIERED_INDEX_ALGO_IVF_PQ = 2
} cuvsTieredIndexANNAlgo;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_TIERED_INDEX_ALGO_CAGRA` | `0` |
| `CUVS_TIERED_INDEX_ALGO_IVF_FLAT` | `1` |
| `CUVS_TIERED_INDEX_ALGO_IVF_PQ` | `2` |

## Tiered Index

<a id="cuvstieredindex"></a>
### cuvsTieredIndex

Struct to hold address of cuvs::neighbors::tiered_index::index and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
  cuvsTieredIndexANNAlgo algo;
} cuvsTieredIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |
| `algo` | [`cuvsTieredIndexANNAlgo`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindexannalgo) |  |

<a id="cuvstieredindexcreate"></a>
### cuvsTieredIndexCreate

Allocate Tiered Index

```c
cuvsError_t cuvsTieredIndexCreate(cuvsTieredIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsTieredIndex_t*`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindex) | cuvsTieredIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvstieredindexdestroy"></a>
### cuvsTieredIndexDestroy

De-allocate Tiered index

```c
cuvsError_t cuvsTieredIndexDestroy(cuvsTieredIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsTieredIndex_t`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindex) | cuvsTieredIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Tiered Index build parameters

<a id="cuvstieredindexparams"></a>
### cuvsTieredIndexParams

Supplemental parameters to build a TieredIndex

```c
struct cuvsTieredIndexParams {
  cuvsDistanceType metric;
  cuvsTieredIndexANNAlgo algo;
  int64_t min_ann_rows;
  bool create_ann_index_on_extend;
  cuvsCagraIndexParams_t cagra_params;
  cuvsIvfFlatIndexParams_t ivf_flat_params;
  cuvsIvfPqIndexParams_t ivf_pq_params;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | Distance type. |
| `algo` | [`cuvsTieredIndexANNAlgo`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindexannalgo) | The type of ANN algorithm we are using |
| `min_ann_rows` | `int64_t` | The minimum number of rows necessary in the index to create an ann index |
| `create_ann_index_on_extend` | `bool` | Whether or not to create a new ann index on extend, if the number of rows in the incremental (bfknn) portion is above min_ann_rows |
| `cagra_params` | [`cuvsCagraIndexParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) | Optional parameters for building a cagra index |
| `ivf_flat_params` | [`cuvsIvfFlatIndexParams_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindexparams) | Optional parameters for building a ivf_flat index |
| `ivf_pq_params` | [`cuvsIvfPqIndexParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindexparams) | Optional parameters for building a ivf-pq index |

<a id="cuvstieredindexparamscreate"></a>
### cuvsTieredIndexParamsCreate

Allocate Tiered Index Params and populate with default values

```c
cuvsError_t cuvsTieredIndexParamsCreate(cuvsTieredIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsTieredIndexParams_t*`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindexparams) | cuvsTieredIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvstieredindexparamsdestroy"></a>
### cuvsTieredIndexParamsDestroy

De-allocate Tiered Index params

```c
cuvsError_t cuvsTieredIndexParamsDestroy(cuvsTieredIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsTieredIndexParams_t`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Tiered index build

<a id="cuvstieredindexbuild"></a>
### cuvsTieredIndexBuild

Build a TieredIndex index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`

```c
cuvsError_t cuvsTieredIndexBuild(cuvsResources_t res,
cuvsTieredIndexParams_t index_params,
DLManagedTensor* dataset,
cuvsTieredIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index_params` | in | [`cuvsTieredIndexParams_t`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindexparams) | Index parameters to use when building the index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | [`cuvsTieredIndex_t`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindex) | cuvsTieredIndex_t Newly built TieredIndex index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Tiered index search

<a id="cuvstieredindexsearch"></a>
### cuvsTieredIndexSearch

Search a TieredIndex index with a `DLManagedTensor`

```c
cuvsError_t cuvsTieredIndexSearch(cuvsResources_t res,
void* search_params,
cuvsTieredIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter prefilter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `search_params` | in | `void*` | params used to the ANN index, should be one of cuvsCagraSearchParams_t, cuvsIvfFlatSearchParams_t, cuvsIvfPqSearchParams_t depending on the type of the tiered index used |
| `index` | in | [`cuvsTieredIndex_t`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindex) | cuvsTieredIndex which has been returned by `cuvsTieredIndexBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `prefilter` | in | [`cuvsFilter`](/api-reference/c-api-neighbors-common#cuvsfilter) | cuvsFilter input prefilter that can be used to filter queries and neighbors based on the given bitmap. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Tiered index extend

<a id="cuvstieredindexextend"></a>
### cuvsTieredIndexExtend

Extend the index with the new data.

```c
cuvsError_t cuvsTieredIndexExtend(cuvsResources_t res,
DLManagedTensor* new_vectors,
cuvsTieredIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `new_vectors` | in | `DLManagedTensor*` | DLManagedTensor* the new vectors to add to the index |
| `index` | inout | [`cuvsTieredIndex_t`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindex) | Tiered index to be extended |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Tiered index merge

<a id="cuvstieredindexmerge"></a>
### cuvsTieredIndexMerge

Merge multiple indices together into a single index

```c
cuvsError_t cuvsTieredIndexMerge(cuvsResources_t res,
cuvsTieredIndexParams_t index_params,
cuvsTieredIndex_t* indices,
size_t num_indices,
cuvsTieredIndex_t output_index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index_params` | in | [`cuvsTieredIndexParams_t`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindexparams) | Index parameters to use when merging |
| `indices` | in | [`cuvsTieredIndex_t*`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindex) | pointers to indices to merge together |
| `num_indices` | in | `size_t` | the number of indices to merge |
| `output_index` | out | [`cuvsTieredIndex_t`](/api-reference/c-api-neighbors-tiered-index#cuvstieredindex) | the merged index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
