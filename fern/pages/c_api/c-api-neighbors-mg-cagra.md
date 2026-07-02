---
slug: api-reference/c-api-neighbors-mg-cagra
---

# Multi-GPU Cagra

_Source header: `cuvs/neighbors/mg_cagra.h`_

## Multi-GPU CAGRA index build parameters

<a id="cuvsmultigpucagraindexparams"></a>
### cuvsMultiGpuCagraIndexParams

Multi-GPU parameters to build CAGRA Index

This structure extends the base CAGRA index parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuCagraIndexParams {
  cuvsCagraIndexParams_t base_params;
  cuvsMultiGpuDistributionMode mode;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | [`cuvsCagraIndexParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagraindexparams) | Base CAGRA index parameters |
| `mode` | [`cuvsMultiGpuDistributionMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpudistributionmode) | Distribution mode for multi-GPU setup |

<a id="cuvsmultigpucagraindexparamscreate"></a>
### cuvsMultiGpuCagraIndexParamsCreate

Allocate Multi-GPU CAGRA Index params, and populate with default values

```c
cuvsError_t cuvsMultiGpuCagraIndexParamsCreate(cuvsMultiGpuCagraIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsMultiGpuCagraIndexParams_t*`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindexparams) | cuvsMultiGpuCagraIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpucagraindexparamsdestroy"></a>
### cuvsMultiGpuCagraIndexParamsDestroy

De-allocate Multi-GPU CAGRA Index params

```c
cuvsError_t cuvsMultiGpuCagraIndexParamsDestroy(cuvsMultiGpuCagraIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsMultiGpuCagraIndexParams_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU CAGRA index search parameters

<a id="cuvsmultigpucagrasearchparams"></a>
### cuvsMultiGpuCagraSearchParams

Multi-GPU parameters to search CAGRA index

This structure extends the base CAGRA search parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuCagraSearchParams {
  cuvsCagraSearchParams_t base_params;
  cuvsMultiGpuReplicatedSearchMode search_mode;
  cuvsMultiGpuShardedMergeMode merge_mode;
  int64_t n_rows_per_batch;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | [`cuvsCagraSearchParams_t`](/api-reference/c-api-neighbors-cagra#cuvscagrasearchparams) | Base CAGRA search parameters |
| `search_mode` | [`cuvsMultiGpuReplicatedSearchMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpureplicatedsearchmode) | Replicated search mode |
| `merge_mode` | [`cuvsMultiGpuShardedMergeMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpushardedmergemode) | Sharded merge mode |
| `n_rows_per_batch` | `int64_t` | Number of rows per batch |

<a id="cuvsmultigpucagrasearchparamscreate"></a>
### cuvsMultiGpuCagraSearchParamsCreate

Allocate Multi-GPU CAGRA search params, and populate with default values

```c
cuvsError_t cuvsMultiGpuCagraSearchParamsCreate(cuvsMultiGpuCagraSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsMultiGpuCagraSearchParams_t*`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagrasearchparams) | cuvsMultiGpuCagraSearchParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpucagrasearchparamsdestroy"></a>
### cuvsMultiGpuCagraSearchParamsDestroy

De-allocate Multi-GPU CAGRA search params

```c
cuvsError_t cuvsMultiGpuCagraSearchParamsDestroy(cuvsMultiGpuCagraSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsMultiGpuCagraSearchParams_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagrasearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU CAGRA index

<a id="cuvsmultigpucagraindex"></a>
### cuvsMultiGpuCagraIndex

Struct to hold address of cuvs::neighbors::mg_index&lt;cagra::index&gt; and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsMultiGpuCagraIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsmultigpucagraindexcreate"></a>
### cuvsMultiGpuCagraIndexCreate

Allocate Multi-GPU CAGRA index

```c
cuvsError_t cuvsMultiGpuCagraIndexCreate(cuvsMultiGpuCagraIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsMultiGpuCagraIndex_t*`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindex) | cuvsMultiGpuCagraIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpucagraindexdestroy"></a>
### cuvsMultiGpuCagraIndexDestroy

De-allocate Multi-GPU CAGRA index

```c
cuvsError_t cuvsMultiGpuCagraIndexDestroy(cuvsMultiGpuCagraIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsMultiGpuCagraIndex_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindex) | cuvsMultiGpuCagraIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU CAGRA index build

<a id="cuvsmultigpucagrabuild"></a>
### cuvsMultiGpuCagraBuild

Build a Multi-GPU CAGRA index

```c
cuvsError_t cuvsMultiGpuCagraBuild(cuvsResources_t res,
cuvsMultiGpuCagraIndexParams_t params,
DLManagedTensor* dataset_tensor,
cuvsMultiGpuCagraIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsMultiGpuCagraIndexParams_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindexparams) | Multi-GPU CAGRA index parameters |
| `dataset_tensor` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | [`cuvsMultiGpuCagraIndex_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindex) | Multi-GPU CAGRA index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU CAGRA index search

<a id="cuvsmultigpucagrasearch"></a>
### cuvsMultiGpuCagraSearch

Search a Multi-GPU CAGRA index

```c
cuvsError_t cuvsMultiGpuCagraSearch(cuvsResources_t res,
cuvsMultiGpuCagraSearchParams_t params,
cuvsMultiGpuCagraIndex_t index,
DLManagedTensor* queries_tensor,
DLManagedTensor* neighbors_tensor,
DLManagedTensor* distances_tensor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsMultiGpuCagraSearchParams_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagrasearchparams) | Multi-GPU CAGRA search parameters |
| `index` | in | [`cuvsMultiGpuCagraIndex_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindex) | Multi-GPU CAGRA index |
| `queries_tensor` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset |
| `neighbors_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output neighbors |
| `distances_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output distances |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU CAGRA index extend

<a id="cuvsmultigpucagraextend"></a>
### cuvsMultiGpuCagraExtend

Extend a Multi-GPU CAGRA index

```c
cuvsError_t cuvsMultiGpuCagraExtend(cuvsResources_t res,
cuvsMultiGpuCagraIndex_t index,
DLManagedTensor* new_vectors_tensor,
DLManagedTensor* new_indices_tensor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index` | in,out | [`cuvsMultiGpuCagraIndex_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindex) | Multi-GPU CAGRA index to extend |
| `new_vectors_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new vectors to add |
| `new_indices_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new indices (optional, can be NULL) |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU CAGRA index serialize

<a id="cuvsmultigpucagraserialize"></a>
### cuvsMultiGpuCagraSerialize

Serialize a Multi-GPU CAGRA index to file

```c
cuvsError_t cuvsMultiGpuCagraSerialize(cuvsResources_t res,
cuvsMultiGpuCagraIndex_t index,
const char* filename);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index` | in | [`cuvsMultiGpuCagraIndex_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindex) | Multi-GPU CAGRA index to serialize |
| `filename` | in | `const char*` | Path to the output file |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU CAGRA index deserialize

<a id="cuvsmultigpucagradeserialize"></a>
### cuvsMultiGpuCagraDeserialize

Deserialize a Multi-GPU CAGRA index from file

```c
cuvsError_t cuvsMultiGpuCagraDeserialize(cuvsResources_t res,
const char* filename,
cuvsMultiGpuCagraIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the input file |
| `index` | out | [`cuvsMultiGpuCagraIndex_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindex) | Multi-GPU CAGRA index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU CAGRA index distribute

<a id="cuvsmultigpucagradistribute"></a>
### cuvsMultiGpuCagraDistribute

Distribute a local CAGRA index to create a Multi-GPU index

```c
cuvsError_t cuvsMultiGpuCagraDistribute(cuvsResources_t res,
const char* filename,
cuvsMultiGpuCagraIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the local index file |
| `index` | out | [`cuvsMultiGpuCagraIndex_t`](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraindex) | Multi-GPU CAGRA index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
