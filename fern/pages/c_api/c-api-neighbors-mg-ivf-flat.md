---
slug: api-reference/c-api-neighbors-mg-ivf-flat
---

# Multi-GPU IVF Flat

_Source header: `cuvs/neighbors/mg_ivf_flat.h`_

## Multi-GPU IVF-Flat index build parameters

<a id="cuvsmultigpuivfflatindexparams"></a>
### cuvsMultiGpuIvfFlatIndexParams

Multi-GPU parameters to build IVF-Flat Index

This structure extends the base IVF-Flat index parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuIvfFlatIndexParams {
  cuvsIvfFlatIndexParams_t base_params;
  cuvsMultiGpuDistributionMode mode;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | [`cuvsIvfFlatIndexParams_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindexparams) | Base IVF-Flat index parameters |
| `mode` | [`cuvsMultiGpuDistributionMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpudistributionmode) | Distribution mode for multi-GPU setup |

<a id="cuvsmultigpuivfflatindexparamscreate"></a>
### cuvsMultiGpuIvfFlatIndexParamsCreate

Allocate Multi-GPU IVF-Flat Index params, and populate with default values

```c
cuvsError_t cuvsMultiGpuIvfFlatIndexParamsCreate(cuvsMultiGpuIvfFlatIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsMultiGpuIvfFlatIndexParams_t*`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindexparams) | cuvsMultiGpuIvfFlatIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpuivfflatindexparamsdestroy"></a>
### cuvsMultiGpuIvfFlatIndexParamsDestroy

De-allocate Multi-GPU IVF-Flat Index params

```c
cuvsError_t cuvsMultiGpuIvfFlatIndexParamsDestroy(cuvsMultiGpuIvfFlatIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsMultiGpuIvfFlatIndexParams_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU IVF-Flat index search parameters

<a id="cuvsmultigpuivfflatsearchparams"></a>
### cuvsMultiGpuIvfFlatSearchParams

Multi-GPU parameters to search IVF-Flat index

This structure extends the base IVF-Flat search parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuIvfFlatSearchParams {
  cuvsIvfFlatSearchParams_t base_params;
  cuvsMultiGpuReplicatedSearchMode search_mode;
  cuvsMultiGpuShardedMergeMode merge_mode;
  int64_t n_rows_per_batch;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | [`cuvsIvfFlatSearchParams_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatsearchparams) | Base IVF-Flat search parameters |
| `search_mode` | [`cuvsMultiGpuReplicatedSearchMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpureplicatedsearchmode) | Replicated search mode |
| `merge_mode` | [`cuvsMultiGpuShardedMergeMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpushardedmergemode) | Sharded merge mode |
| `n_rows_per_batch` | `int64_t` | Number of rows per batch |

<a id="cuvsmultigpuivfflatsearchparamscreate"></a>
### cuvsMultiGpuIvfFlatSearchParamsCreate

Allocate Multi-GPU IVF-Flat search params, and populate with default values

```c
cuvsError_t cuvsMultiGpuIvfFlatSearchParamsCreate(cuvsMultiGpuIvfFlatSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsMultiGpuIvfFlatSearchParams_t*`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatsearchparams) | cuvsMultiGpuIvfFlatSearchParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpuivfflatsearchparamsdestroy"></a>
### cuvsMultiGpuIvfFlatSearchParamsDestroy

De-allocate Multi-GPU IVF-Flat search params

```c
cuvsError_t cuvsMultiGpuIvfFlatSearchParamsDestroy(cuvsMultiGpuIvfFlatSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsMultiGpuIvfFlatSearchParams_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatsearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU IVF-Flat index

<a id="cuvsmultigpuivfflatindex"></a>
### cuvsMultiGpuIvfFlatIndex

Struct to hold address of cuvs::neighbors::mg_index&lt;ivf_flat::index&gt; and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsMultiGpuIvfFlatIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsmultigpuivfflatindexcreate"></a>
### cuvsMultiGpuIvfFlatIndexCreate

Allocate Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatIndexCreate(cuvsMultiGpuIvfFlatIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsMultiGpuIvfFlatIndex_t*`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindex) | cuvsMultiGpuIvfFlatIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsmultigpuivfflatindexdestroy"></a>
### cuvsMultiGpuIvfFlatIndexDestroy

De-allocate Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatIndexDestroy(cuvsMultiGpuIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsMultiGpuIvfFlatIndex_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindex) | cuvsMultiGpuIvfFlatIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU IVF-Flat index build

<a id="cuvsmultigpuivfflatbuild"></a>
### cuvsMultiGpuIvfFlatBuild

Build a Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatBuild(cuvsResources_t res,
cuvsMultiGpuIvfFlatIndexParams_t params,
DLManagedTensor* dataset_tensor,
cuvsMultiGpuIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsMultiGpuIvfFlatIndexParams_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindexparams) | Multi-GPU IVF-Flat index parameters |
| `dataset_tensor` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | [`cuvsMultiGpuIvfFlatIndex_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindex) | Multi-GPU IVF-Flat index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU IVF-Flat index search

<a id="cuvsmultigpuivfflatsearch"></a>
### cuvsMultiGpuIvfFlatSearch

Search a Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatSearch(cuvsResources_t res,
cuvsMultiGpuIvfFlatSearchParams_t params,
cuvsMultiGpuIvfFlatIndex_t index,
DLManagedTensor* queries_tensor,
DLManagedTensor* neighbors_tensor,
DLManagedTensor* distances_tensor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsMultiGpuIvfFlatSearchParams_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatsearchparams) | Multi-GPU IVF-Flat search parameters |
| `index` | in | [`cuvsMultiGpuIvfFlatIndex_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindex) | Multi-GPU IVF-Flat index |
| `queries_tensor` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset |
| `neighbors_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output neighbors |
| `distances_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output distances |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU IVF-Flat index extend

<a id="cuvsmultigpuivfflatextend"></a>
### cuvsMultiGpuIvfFlatExtend

Extend a Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatExtend(cuvsResources_t res,
cuvsMultiGpuIvfFlatIndex_t index,
DLManagedTensor* new_vectors_tensor,
DLManagedTensor* new_indices_tensor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index` | in,out | [`cuvsMultiGpuIvfFlatIndex_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindex) | Multi-GPU IVF-Flat index to extend |
| `new_vectors_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new vectors to add |
| `new_indices_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new indices (optional, can be NULL) |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU IVF-Flat index serialize

<a id="cuvsmultigpuivfflatserialize"></a>
### cuvsMultiGpuIvfFlatSerialize

Serialize a Multi-GPU IVF-Flat index to file

```c
cuvsError_t cuvsMultiGpuIvfFlatSerialize(cuvsResources_t res,
cuvsMultiGpuIvfFlatIndex_t index,
const char* filename);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index` | in | [`cuvsMultiGpuIvfFlatIndex_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindex) | Multi-GPU IVF-Flat index to serialize |
| `filename` | in | `const char*` | Path to the output file |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU IVF-Flat index deserialize

<a id="cuvsmultigpuivfflatdeserialize"></a>
### cuvsMultiGpuIvfFlatDeserialize

Deserialize a Multi-GPU IVF-Flat index from file

```c
cuvsError_t cuvsMultiGpuIvfFlatDeserialize(cuvsResources_t res,
const char* filename,
cuvsMultiGpuIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the input file |
| `index` | out | [`cuvsMultiGpuIvfFlatIndex_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindex) | Multi-GPU IVF-Flat index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Multi-GPU IVF-Flat index distribute

<a id="cuvsmultigpuivfflatdistribute"></a>
### cuvsMultiGpuIvfFlatDistribute

Distribute a local IVF-Flat index to create a Multi-GPU index

```c
cuvsError_t cuvsMultiGpuIvfFlatDistribute(cuvsResources_t res,
const char* filename,
cuvsMultiGpuIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the local index file |
| `index` | out | [`cuvsMultiGpuIvfFlatIndex_t`](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatindex) | Multi-GPU IVF-Flat index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
