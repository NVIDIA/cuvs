---
slug: api-reference/c-api-neighbors-nn-descent
---

# NN Descent

_Source header: `cuvs/neighbors/nn_descent.h`_

## Types

<a id="cuvsnndescentdistcompdtype"></a>
### cuvsNNDescentDistCompDtype

Dtype to use for distance computation

```c
typedef enum {
  NND_DIST_COMP_AUTO = 0,
  NND_DIST_COMP_FP32 = 1,
  NND_DIST_COMP_FP16 = 2
} cuvsNNDescentDistCompDtype;
```

**Values**

| Name | Value | Description |
| --- | --- | --- |
| `NND_DIST_COMP_AUTO` | `0` | Automatically determine the best dtype for distance computation based on the dataset dimensions. |
| `NND_DIST_COMP_FP32` | `1` | Use fp32 distance computation for better precision at the cost of performance and memory usage. |
| `NND_DIST_COMP_FP16` | `2` | Use fp16 distance computation. |

## The nn-descent algorithm parameters.

<a id="cuvsnndescentindexparams"></a>
### cuvsNNDescentIndexParams

Parameters used to build an nn-descent index

```c
struct cuvsNNDescentIndexParams {
  cuvsDistanceType metric;
  float metric_arg;
  size_t graph_degree;
  size_t intermediate_graph_degree;
  size_t max_iterations;
  float termination_threshold;
  bool return_distances;
  cuvsNNDescentDistCompDtype dist_comp_dtype;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | The distance metric to use |
| `metric_arg` | `float` | The argument used by distance metrics like Minkowskidistance |
| `graph_degree` | `size_t` | For an input dataset of dimensions (N, D), determines the final dimensions of the all-neighbors knn graph which turns out to be of dimensions (N, graph_degree) |
| `intermediate_graph_degree` | `size_t` | Internally, nn-descent builds an all-neighbors knn graph of dimensions (N, intermediate_graph_degree) before selecting the final `graph_degree` neighbors. It's recommended that `intermediate_graph_degree` &gt;= 1.5 * graph_degree |
| `max_iterations` | `size_t` | The number of iterations that nn-descent will refine the graph for. More iterations produce a better quality graph at cost of performance |
| `termination_threshold` | `float` | The delta at which nn-descent will terminate its iterations |
| `return_distances` | `bool` | Boolean to decide whether to return distances array |
| `dist_comp_dtype` | [`cuvsNNDescentDistCompDtype`](/api-reference/c-api-neighbors-nn-descent#cuvsnndescentdistcompdtype) | dtype to use for distance computation.<br />Defaults to `NND_DIST_COMP_AUTO` which automatically determines the best dtype for distance computation based on the dataset dimensions.<br />Use `NND_DIST_COMP_FP32` for better precision at the cost of performance and memory usage. This option is only valid when data type is fp32.<br />Use `NND_DIST_COMP_FP16` for better performance and memory usage at the cost of precision. |

<a id="cuvsnndescentindexparamscreate"></a>
### cuvsNNDescentIndexParamsCreate

Allocate NN-Descent Index params, and populate with default values

```c
cuvsError_t cuvsNNDescentIndexParamsCreate(cuvsNNDescentIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsNNDescentIndexParams_t*`](/api-reference/c-api-neighbors-nn-descent#cuvsnndescentindexparams) | cuvsNNDescentIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsnndescentindexparamsdestroy"></a>
### cuvsNNDescentIndexParamsDestroy

De-allocate NN-Descent Index params

```c
cuvsError_t cuvsNNDescentIndexParamsDestroy(cuvsNNDescentIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsNNDescentIndexParams_t`](/api-reference/c-api-neighbors-nn-descent#cuvsnndescentindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## NN-Descent index

<a id="cuvsnndescentindex"></a>
### cuvsNNDescentIndex

Struct to hold address of cuvs::neighbors::nn_descent::index and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsNNDescentIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsnndescentindexcreate"></a>
### cuvsNNDescentIndexCreate

Allocate NN-Descent index

```c
cuvsError_t cuvsNNDescentIndexCreate(cuvsNNDescentIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsNNDescentIndex_t*`](/api-reference/c-api-neighbors-nn-descent#cuvsnndescentindex) | cuvsNNDescentIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsnndescentindexdestroy"></a>
### cuvsNNDescentIndexDestroy

De-allocate NN-Descent index

```c
cuvsError_t cuvsNNDescentIndexDestroy(cuvsNNDescentIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsNNDescentIndex_t`](/api-reference/c-api-neighbors-nn-descent#cuvsnndescentindex) | cuvsNNDescentIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## NN-Descent index build

<a id="cuvsnndescentbuild"></a>
### cuvsNNDescentBuild

Build a NN-Descent index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`

```c
cuvsError_t cuvsNNDescentBuild(cuvsResources_t res,
cuvsNNDescentIndexParams_t index_params,
DLManagedTensor* dataset,
DLManagedTensor* graph,
cuvsNNDescentIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index_params` | in | [`cuvsNNDescentIndexParams_t`](/api-reference/c-api-neighbors-nn-descent#cuvsnndescentindexparams) | cuvsNNDescentIndexParams_t used to build NN-Descent index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset on host or device memory |
| `graph` | inout | `DLManagedTensor*` | Optional preallocated graph on host memory to store output |
| `index` | out | [`cuvsNNDescentIndex_t`](/api-reference/c-api-neighbors-nn-descent#cuvsnndescentindex) | cuvsNNDescentIndex_t Newly built NN-Descent index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
