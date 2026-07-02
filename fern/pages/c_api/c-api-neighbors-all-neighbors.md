---
slug: api-reference/c-api-neighbors-all-neighbors
---

# All Neighbors

_Source header: `cuvs/neighbors/all_neighbors.h`_

## All-neighbors C-API build parameters

<a id="cuvsallneighborsalgo"></a>
### cuvsAllNeighborsAlgo

Graph build algorithm selection.

```c
typedef enum {
  CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE = 0,
  CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ = 1,
  CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT = 2
} cuvsAllNeighborsAlgo;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE` | `0` |
| `CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ` | `1` |
| `CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT` | `2` |

<a id="cuvsallneighborsindexparams"></a>
### cuvsAllNeighborsIndexParams

Parameters controlling SNMG all-neighbors build.

```c
struct cuvsAllNeighborsIndexParams {
  cuvsAllNeighborsAlgo algo;
  size_t overlap_factor;
  size_t n_clusters;
  cuvsDistanceType metric;
  cuvsIvfPqIndexParams_t ivf_pq_params;
  cuvsNNDescentIndexParams_t nn_descent_params;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `algo` | [`cuvsAllNeighborsAlgo`](/api-reference/c-api-neighbors-all-neighbors#cuvsallneighborsalgo) |  |
| `overlap_factor` | `size_t` |  |
| `n_clusters` | `size_t` |  |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) |  |
| `ivf_pq_params` | [`cuvsIvfPqIndexParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindexparams) |  |
| `nn_descent_params` | [`cuvsNNDescentIndexParams_t`](/api-reference/c-api-neighbors-nn-descent#cuvsnndescentindexparams) |  |

<a id="cuvsallneighborsindexparamscreate"></a>
### cuvsAllNeighborsIndexParamsCreate

Create a default all-neighbors index parameters struct.

```c
cuvsError_t cuvsAllNeighborsIndexParamsCreate(cuvsAllNeighborsIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | out | [`cuvsAllNeighborsIndexParams_t*`](/api-reference/c-api-neighbors-all-neighbors#cuvsallneighborsindexparams) | Pointer to allocated index_params struct |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsallneighborsindexparamsdestroy"></a>
### cuvsAllNeighborsIndexParamsDestroy

Destroy an all-neighbors index parameters struct.

```c
cuvsError_t cuvsAllNeighborsIndexParamsDestroy(cuvsAllNeighborsIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsAllNeighborsIndexParams_t`](/api-reference/c-api-neighbors-all-neighbors#cuvsallneighborsindexparams) | Index parameters struct to destroy |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## All-neighbors C-API build

<a id="cuvsallneighborsbuild"></a>
### cuvsAllNeighborsBuild

Build an all-neighbors k-NN graph automatically detecting host vs device dataset.

```c
cuvsError_t cuvsAllNeighborsBuild(cuvsResources_t res,
cuvsAllNeighborsIndexParams_t params,
DLManagedTensor* dataset,
DLManagedTensor* indices,
DLManagedTensor* distances,
DLManagedTensor* core_distances,
float alpha);
```

The function automatically detects whether the dataset is host-resident or device-resident and calls the appropriate implementation. For host datasets, it partitions data into `n_clusters` clusters and assigns each row to `overlap_factor` nearest clusters. For device datasets, `n_clusters` must be 1 (no batching); `overlap_factor` is ignored. Outputs always reside in device memory.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | Can be a SNMG multi-GPU resources (`cuvsResources_t`) or single-GPU resources |
| `params` | in | [`cuvsAllNeighborsIndexParams_t`](/api-reference/c-api-neighbors-all-neighbors#cuvsallneighborsindexparams) | Build parameters (see cuvsAllNeighborsIndexParams) |
| `dataset` | in | `DLManagedTensor*` | 2D tensor [num_rows x dim] on host or device (auto-detected) |
| `indices` | out | `DLManagedTensor*` | 2D tensor [num_rows x k] on device (int64) |
| `distances` | out | `DLManagedTensor*` | Optional 2D tensor [num_rows x k] on device (float32); can be NULL |
| `core_distances` | out | `DLManagedTensor*` | Optional 1D tensor [num_rows] on device (float32); can be NULL |
| `alpha` | in | `float` | Mutual-reachability scaling; used only when core_distances is provided |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
