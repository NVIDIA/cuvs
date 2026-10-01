---
slug: api-reference/c-api-neighbors-refine
---

# Refine

_Source header: `cuvs/neighbors/refine.h`_

## Approximate Nearest Neighbors Refinement C-API

<a id="cuvsrefine"></a>
### cuvsRefine

Refine nearest neighbor search.

```c
cuvsError_t cuvsRefine(cuvsResources_t res,
DLManagedTensor* dataset,
DLManagedTensor* queries,
DLManagedTensor* candidates,
cuvsDistanceType metric,
DLManagedTensor* indices,
DLManagedTensor* distances);
```

Refinement is an operation that follows an approximate NN search. The approximate search has already selected n_candidates neighbor candidates for each query. We narrow it down to k neighbors. For each query, we calculate the exact distance between the query and its n_candidates neighbor candidate, and select the k nearest ones.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `dataset` | in | `DLManagedTensor*` | device matrix that stores the dataset [n_rows, dims] |
| `queries` | in | `DLManagedTensor*` | device matrix of the queries [n_queris, dims] |
| `candidates` | in | `DLManagedTensor*` | indices of candidate vectors [n_queries, n_candidates], where n_candidates &gt;= k |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | distance metric to use. Euclidean (L2) is used by default |
| `indices` | out | `DLManagedTensor*` | device matrix that stores the refined indices [n_queries, k] |
| `distances` | out | `DLManagedTensor*` | device matrix that stores the refined distances [n_queries, k] |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
