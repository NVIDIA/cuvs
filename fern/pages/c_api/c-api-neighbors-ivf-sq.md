---
slug: api-reference/c-api-neighbors-ivf-sq
---

# IVF SQ

_Source header: `cuvs/neighbors/ivf_sq.h`_

## IVF-SQ index build parameters

<a id="cuvsivfsqindexparams"></a>
### cuvsIvfSqIndexParams

Supplemental parameters to build IVF-SQ Index

```c
struct cuvsIvfSqIndexParams {
  cuvsDistanceType metric;
  float metric_arg;
  bool add_data_on_build;
  uint32_t n_lists;
  uint32_t kmeans_n_iters;
  uint32_t max_train_points_per_cluster;
  bool conservative_memory_allocation;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | Distance type. |
| `metric_arg` | `float` | The argument used by some distance metrics. |
| `add_data_on_build` | `bool` | Whether to add the dataset content to the index, i.e.:<br /><br />- `true` means the index is filled with the dataset vectors and ready to search after calling `build`.<br />- `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but the index is left empty; you'd need to call `extend` on the index afterwards to populate it. |
| `n_lists` | `uint32_t` | The number of inverted lists (clusters) |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (index building). |
| `max_train_points_per_cluster` | `uint32_t` | The number of data vectors per cluster to use during iterative kmeans building. The index uses at most `n_lists * max_train_points_per_cluster` rows for training. |
| `conservative_memory_allocation` | `bool` | By default, the algorithm allocates more space than necessary for individual clusters (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of data copies during repeated calls to `extend` (extending the database).<br /><br />The alternative is the conservative allocation behavior; when enabled, the algorithm always allocates the minimum amount of memory required to store the given number of records. Set this flag to `true` if you prefer to use as little GPU memory for the database as possible. |

<a id="cuvsivfsqindexparamscreate"></a>
### cuvsIvfSqIndexParamsCreate

Allocate IVF-SQ Index params, and populate with default values

```c
cuvsError_t cuvsIvfSqIndexParamsCreate(cuvsIvfSqIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsIvfSqIndexParams_t*`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindexparams) | cuvsIvfSqIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfsqindexparamsdestroy"></a>
### cuvsIvfSqIndexParamsDestroy

De-allocate IVF-SQ Index params

```c
cuvsError_t cuvsIvfSqIndexParamsDestroy(cuvsIvfSqIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsIvfSqIndexParams_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-SQ index search parameters

<a id="cuvsivfsqsearchparams"></a>
### cuvsIvfSqSearchParams

Supplemental parameters to search IVF-SQ index

```c
struct cuvsIvfSqSearchParams {
  uint32_t n_probes;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |

<a id="cuvsivfsqsearchparamscreate"></a>
### cuvsIvfSqSearchParamsCreate

Allocate IVF-SQ search params, and populate with default values

```c
cuvsError_t cuvsIvfSqSearchParamsCreate(cuvsIvfSqSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsIvfSqSearchParams_t*`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqsearchparams) | cuvsIvfSqSearchParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfsqsearchparamsdestroy"></a>
### cuvsIvfSqSearchParamsDestroy

De-allocate IVF-SQ search params

```c
cuvsError_t cuvsIvfSqSearchParamsDestroy(cuvsIvfSqSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsIvfSqSearchParams_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqsearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-SQ index

<a id="cuvsivfsqindex"></a>
### cuvsIvfSqIndex

Struct to hold address of cuvs::neighbors::ivf_sq::index and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsIvfSqIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsivfsqindexcreate"></a>
### cuvsIvfSqIndexCreate

Allocate IVF-SQ index

```c
cuvsError_t cuvsIvfSqIndexCreate(cuvsIvfSqIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfSqIndex_t*`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) | cuvsIvfSqIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfsqindexdestroy"></a>
### cuvsIvfSqIndexDestroy

De-allocate IVF-SQ index

```c
cuvsError_t cuvsIvfSqIndexDestroy(cuvsIvfSqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) | cuvsIvfSqIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfsqindexgetnlists"></a>
### cuvsIvfSqIndexGetNLists

Get the number of clusters/inverted lists

```c
cuvsError_t cuvsIvfSqIndexGetNLists(cuvsIvfSqIndex_t index, int64_t* n_lists);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) |  |
| `n_lists` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfsqindexgetdim"></a>
### cuvsIvfSqIndexGetDim

Get the dimensionality of the data

```c
cuvsError_t cuvsIvfSqIndexGetDim(cuvsIvfSqIndex_t index, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) |  |
| `dim` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfsqindexgetsize"></a>
### cuvsIvfSqIndexGetSize

Get the size of the index

```c
cuvsError_t cuvsIvfSqIndexGetSize(cuvsIvfSqIndex_t index, int64_t* size);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) |  |
| `size` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfsqindexgetcenters"></a>
### cuvsIvfSqIndexGetCenters

Get the cluster centers corresponding to the lists [n_lists, dim]

```c
cuvsError_t cuvsIvfSqIndexGetCenters(cuvsIvfSqIndex_t index, DLManagedTensor* centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) | cuvsIvfSqIndex_t Built Ivf-SQ Index |
| `centers` | out | `DLManagedTensor*` | Preallocated array on host or device memory to store output, [n_lists, dim] |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-SQ index build

<a id="cuvsivfsqbuild"></a>
### cuvsIvfSqBuild

Build an IVF-SQ index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`

```c
cuvsError_t cuvsIvfSqBuild(cuvsResources_t res,
cuvsIvfSqIndexParams_t index_params,
DLManagedTensor* dataset,
cuvsIvfSqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index_params` | in | [`cuvsIvfSqIndexParams_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindexparams) | cuvsIvfSqIndexParams_t used to build IVF-SQ index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) | cuvsIvfSqIndex_t Newly built IVF-SQ index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-SQ index search

<a id="cuvsivfsqsearch"></a>
### cuvsIvfSqSearch

Search an IVF-SQ index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. Types for input are:

1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` or 16
2. `neighbors`: `kDLDataType.code == kDLInt` and `kDLDataType.bits = 64`
3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

```c
cuvsError_t cuvsIvfSqSearch(cuvsResources_t res,
cuvsIvfSqSearchParams_t search_params,
cuvsIvfSqIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter filter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `search_params` | in | [`cuvsIvfSqSearchParams_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqsearchparams) | cuvsIvfSqSearchParams_t used to search IVF-SQ index |
| `index` | in | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) | ivfSqIndex which has been returned by `cuvsIvfSqBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `filter` | in | [`cuvsFilter`](/api-reference/c-api-neighbors-common#cuvsfilter) | cuvsFilter input filter that can be used to filter queries and neighbors based on the given bitset. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-SQ C-API serialize functions

<a id="cuvsivfsqserialize"></a>
### cuvsIvfSqSerialize

Save the index to file.

```c
cuvsError_t cuvsIvfSqSerialize(cuvsResources_t res, const char* filename, cuvsIvfSqIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) | IVF-SQ index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfsqdeserialize"></a>
### cuvsIvfSqDeserialize

Load index from file.

```c
cuvsError_t cuvsIvfSqDeserialize(cuvsResources_t res,
const char* filename,
cuvsIvfSqIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | out | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) | IVF-SQ index loaded from disk |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-SQ index extend

<a id="cuvsivfsqextend"></a>
### cuvsIvfSqExtend

Extend the index with the new data.

```c
cuvsError_t cuvsIvfSqExtend(cuvsResources_t res,
DLManagedTensor* new_vectors,
DLManagedTensor* new_indices,
cuvsIvfSqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `new_vectors` | in | `DLManagedTensor*` | DLManagedTensor* the new vectors to add to the index |
| `new_indices` | in | `DLManagedTensor*` | DLManagedTensor* vector of new indices for the new vectors. If the index is empty, this can be NULL to imply a continuous range `[0...n_rows)`. |
| `index` | inout | [`cuvsIvfSqIndex_t`](/api-reference/c-api-neighbors-ivf-sq#cuvsivfsqindex) | IVF-SQ index to be extended |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
