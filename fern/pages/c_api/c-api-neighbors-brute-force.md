---
slug: api-reference/c-api-neighbors-brute-force
---

# Brute Force

_Source header: `cuvs/neighbors/brute_force.h`_

## Bruteforce index

<a id="cuvsbruteforceindex"></a>
### cuvsBruteForceIndex

Struct to hold address of cuvs::neighbors::brute_force::index and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsBruteForceIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsbruteforceindexcreate"></a>
### cuvsBruteForceIndexCreate

Allocate BRUTEFORCE index

```c
cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsBruteForceIndex_t*`](/api-reference/c-api-neighbors-brute-force#cuvsbruteforceindex) | cuvsBruteForceIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbruteforceindexdestroy"></a>
### cuvsBruteForceIndexDestroy

De-allocate BRUTEFORCE index

```c
cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsBruteForceIndex_t`](/api-reference/c-api-neighbors-brute-force#cuvsbruteforceindex) | cuvsBruteForceIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Bruteforce index build

<a id="cuvsbruteforcebuild"></a>
### cuvsBruteForceBuild

Build a BRUTEFORCE index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`

```c
cuvsError_t cuvsBruteForceBuild(cuvsResources_t res,
DLManagedTensor* dataset,
cuvsDistanceType metric,
float metric_arg,
cuvsBruteForceIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | metric |
| `metric_arg` | in | `float` | metric_arg |
| `index` | out | [`cuvsBruteForceIndex_t`](/api-reference/c-api-neighbors-brute-force#cuvsbruteforceindex) | cuvsBruteForceIndex_t Newly built BRUTEFORCE index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## Bruteforce index search

<a id="cuvsbruteforcesearch"></a>
### cuvsBruteForceSearch

Search a BRUTEFORCE index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. It is also important to note that the BRUTEFORCE index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Types for input are:

1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` or `kDLDataType.bits = 16`
2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

```c
cuvsError_t cuvsBruteForceSearch(cuvsResources_t res,
cuvsBruteForceIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter prefilter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index` | in | [`cuvsBruteForceIndex_t`](/api-reference/c-api-neighbors-brute-force#cuvsbruteforceindex) | cuvsBruteForceIndex which has been returned by `cuvsBruteForceBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `prefilter` | in | [`cuvsFilter`](/api-reference/c-api-neighbors-common#cuvsfilter) | cuvsFilter input prefilter that can be used to filter queries and neighbors based on the given bitmap. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## BRUTEFORCE C-API serialize functions

<a id="cuvsbruteforceserialize"></a>
### cuvsBruteForceSerialize

Save the index to file.

```c
cuvsError_t cuvsBruteForceSerialize(cuvsResources_t res,
const char* filename,
cuvsBruteForceIndex_t index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | [`cuvsBruteForceIndex_t`](/api-reference/c-api-neighbors-brute-force#cuvsbruteforceindex) | BRUTEFORCE index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbruteforcedeserialize"></a>
### cuvsBruteForceDeserialize

Load index from file.

```c
cuvsError_t cuvsBruteForceDeserialize(cuvsResources_t res,
const char* filename,
cuvsBruteForceIndex_t index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | out | [`cuvsBruteForceIndex_t`](/api-reference/c-api-neighbors-brute-force#cuvsbruteforceindex) | BRUTEFORCE index loaded disk |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
