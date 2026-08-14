---
slug: api-reference/c-api-neighbors-ivf-flat
---

# IVF Flat

_Source header: `cuvs/neighbors/ivf_flat.h`_

## IVF-Flat index build parameters

<a id="cuvsivfflatindexparams"></a>
### cuvsIvfFlatIndexParams

Supplemental parameters to build IVF-Flat Index

```c
struct cuvsIvfFlatIndexParams {
  cuvsDistanceType metric;
  float metric_arg;
  bool add_data_on_build;
  uint32_t n_lists;
  uint32_t kmeans_n_iters;
  double kmeans_trainset_fraction;
  bool adaptive_centers;
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
| `kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building. |
| `adaptive_centers` | `bool` | By default (adaptive_centers = false), the cluster centers are trained in `ivf_flat::build`, and never modified in `ivf_flat::extend`. As a result, you may need to retrain the index from scratch after invoking (`ivf_flat::extend`) a few times with new data, the distribution of which is no longer representative of the original training set.<br /><br />The alternative behavior (adaptive_centers = true) is to update the cluster centers for new data when it is added. In this case, `index.centers()` are always exactly the centroids of the data in the corresponding clusters. The drawback of this behavior is that the centroids depend on the order of adding new data (through the classification of the added data); that is, `index.centers()` "drift" together with the changing distribution of the newly added data. |
| `conservative_memory_allocation` | `bool` | By default, the algorithm allocates more space than necessary for individual clusters (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of data copies during repeated calls to `extend` (extending the database).<br /><br />The alternative is the conservative allocation behavior; when enabled, the algorithm always allocates the minimum amount of memory required to store the given number of records. Set this flag to `true` if you prefer to use as little GPU memory for the database as possible. |

<a id="cuvsivfflatindexparamscreate"></a>
### cuvsIvfFlatIndexParamsCreate

Allocate IVF-Flat Index params, and populate with default values

```c
cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsIvfFlatIndexParams_t*`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindexparams) | cuvsIvfFlatIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfflatindexparamsdestroy"></a>
### cuvsIvfFlatIndexParamsDestroy

De-allocate IVF-Flat Index params

```c
cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsIvfFlatIndexParams_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-Flat index search parameters

<a id="cuvsivfflatsearchparams"></a>
### cuvsIvfFlatSearchParams

Supplemental parameters to search IVF-Flat index

```c
struct cuvsIvfFlatSearchParams {
  uint32_t n_probes;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |

<a id="cuvsivfflatsearchparamscreate"></a>
### cuvsIvfFlatSearchParamsCreate

Allocate IVF-Flat search params, and populate with default values

```c
cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsIvfFlatSearchParams_t*`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatsearchparams) | cuvsIvfFlatSearchParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfflatsearchparamsdestroy"></a>
### cuvsIvfFlatSearchParamsDestroy

De-allocate IVF-Flat search params

```c
cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsIvfFlatSearchParams_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatsearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-Flat index

<a id="cuvsivfflatindex"></a>
### cuvsIvfFlatIndex

Struct to hold address of cuvs::neighbors::ivf_flat::index and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsIvfFlatIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsivfflatindexcreate"></a>
### cuvsIvfFlatIndexCreate

Allocate IVF-Flat index

```c
cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfFlatIndex_t*`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | cuvsIvfFlatIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfflatindexdestroy"></a>
### cuvsIvfFlatIndexDestroy

De-allocate IVF-Flat index

```c
cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | cuvsIvfFlatIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfflatindexgetnlists"></a>
### cuvsIvfFlatIndexGetNLists

Get the number of clusters/inverted lists in the index

```c
cuvsError_t cuvsIvfFlatIndexGetNLists(cuvsIvfFlatIndex_t index, int64_t* n_lists);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | cuvsIvfFlatIndex_t Built IVF-Flat index |
| `n_lists` | out | `int64_t*` | Pointer to store the number of lists |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfflatindexgetdim"></a>
### cuvsIvfFlatIndexGetDim

Get the dimensionality of the indexed data

```c
cuvsError_t cuvsIvfFlatIndexGetDim(cuvsIvfFlatIndex_t index, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | cuvsIvfFlatIndex_t Built IVF-Flat index |
| `dim` | out | `int64_t*` | Pointer to store the dimensionality |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfflatindexgetcenters"></a>
### cuvsIvfFlatIndexGetCenters

Get the cluster centers corresponding to the lists [n_lists, dim]

```c
cuvsError_t cuvsIvfFlatIndexGetCenters(cuvsIvfFlatIndex_t index, DLManagedTensor* centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | cuvsIvfFlatIndex_t Built Ivf-Flat Index |
| `centers` | out | `DLManagedTensor*` | Preallocated array on host or device memory to store output, [n_lists, dim] |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-Flat index build

<a id="cuvsivfflatbuild"></a>
### cuvsIvfFlatBuild

Build a IVF-Flat index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
3. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`

```c
cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res,
cuvsIvfFlatIndexParams_t index_params,
DLManagedTensor* dataset,
cuvsIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index_params` | in | [`cuvsIvfFlatIndexParams_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindexparams) | cuvsIvfFlatIndexParams_t used to build IVF-Flat index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | cuvsIvfFlatIndex_t Newly built IVF-Flat index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-Flat index search

<a id="cuvsivfflatsearch"></a>
### cuvsIvfFlatSearch

Search a IVF-Flat index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. It is also important to note that the IVF-Flat Index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Types for input are:

1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

```c
cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res,
cuvsIvfFlatSearchParams_t search_params,
cuvsIvfFlatIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter filter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `search_params` | in | [`cuvsIvfFlatSearchParams_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatsearchparams) | cuvsIvfFlatSearchParams_t used to search IVF-Flat index |
| `index` | in | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | ivfFlatIndex which has been returned by `ivfFlatBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `filter` | in | [`cuvsFilter`](/api-reference/c-api-neighbors-common#cuvsfilter) | cuvsFilter input filter that can be used to filter queries and neighbors based on the given bitset. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-Flat C-API serialize functions

<a id="cuvsivfflatserialize"></a>
### cuvsIvfFlatSerialize

Save the index to file.

```c
cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res,
const char* filename,
cuvsIvfFlatIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | IVF-Flat index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfflatdeserialize"></a>
### cuvsIvfFlatDeserialize

Load index from file.

```c
cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res,
const char* filename,
cuvsIvfFlatIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | out | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | IVF-Flat index loaded disk |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-Flat index extend

<a id="cuvsivfflatextend"></a>
### cuvsIvfFlatExtend

Extend the index with the new data.

```c
cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res,
DLManagedTensor* new_vectors,
DLManagedTensor* new_indices,
cuvsIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `new_vectors` | in | `DLManagedTensor*` | DLManagedTensor* the new vectors to add to the index |
| `new_indices` | in | `DLManagedTensor*` | DLManagedTensor* vector of new indices for the new vectors |
| `index` | inout | [`cuvsIvfFlatIndex_t`](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatindex) | IVF-Flat index to be extended |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
