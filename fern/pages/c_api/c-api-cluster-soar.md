---
slug: api-reference/c-api-cluster-soar
---

# Soar

_Source header: `cuvs/cluster/soar.h`_

## SOAR hyperparameters

<a id="cuvssoarparams"></a>
### cuvsSoarParams

Hyper-parameters for SOAR assignment.

```c
struct cuvsSoarParams {
  float lambda;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `lambda` | `float` | Weight of the projection of the secondary residual onto the primary residual in the SOAR loss. Larger values penalize secondary centroids whose residual is aligned with the primary residual, favoring complementary assignments. `0` reduces the loss to plain squared distance, which the primary centroid itself minimizes, so nothing is spilled. Default: 1.0. |

<a id="cuvssoarparamscreate"></a>
### cuvsSoarParamsCreate

Allocate SOAR params, and populate with default values

```c
cuvsError_t cuvsSoarParamsCreate(cuvsSoarParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | out | [`cuvsSoarParams_t*`](/api-reference/c-api-cluster-soar#cuvssoarparams) | cuvsSoarParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvssoarparamsdestroy"></a>
### cuvsSoarParamsDestroy

De-allocate SOAR params

```c
cuvsError_t cuvsSoarParamsDestroy(cuvsSoarParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsSoarParams_t`](/api-reference/c-api-cluster-soar#cuvssoarparams) | cuvsSoarParams_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## SOAR assignment

<a id="cuvssoarpredict"></a>
### cuvsSoarPredict

Assign a secondary ("spilled") cluster to each row of the dataset.

```c
cuvsError_t cuvsSoarPredict(cuvsResources_t res,
cuvsSoarParams_t params,
DLManagedTensor* dataset,
DLManagedTensor* centroids,
DLManagedTensor* labels,
DLManagedTensor* soar_labels);
```

SOAR (Spilling with Orthogonality-Amplified Residuals) picks, for each vector, a second centroid that complements the primary assignment instead of merely being the next-closest one. It minimizes the loss of Theorem 3.1 of https://arxiv.org/abs/2404.00774: for a vector `x` with primary residual `r = x - centroids[labels[i]]`,

`score(c) = \|\|x - c\|\|^2 + lambda * (dot(r / \|\|r\|\|, x - c))^2`

and `soar_labels[i]` is the centroid minimizing that score. Indexing a vector under both its primary and its secondary centroid improves recall for queries near a partition boundary.

All tensors must be on device memory. `dataset` and `centroids` must be row-major float32. `labels` and `soar_labels` must have the same dtype, either uint32 or int32; int32 is accepted so that the output of `cuvsKMeansPredict` can be passed through without a conversion.

The primary centroid is not excluded from the search, so `soar_labels[i] == labels[i]` is a possible (and meaningful) result: it says that no other centroid is worth spilling to, which is the common case for vectors in the interior of a cluster.

Scratch memory scales as `n_rows * n_clusters * 4` bytes because scores against all centroids are materialized at once and are not tiled. Process the dataset in row batches to bound the peak device memory usage.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | opaque C handle |
| `params` | in | [`cuvsSoarParams_t`](/api-reference/c-api-cluster-soar#cuvssoarparams) | Parameters for SOAR assignment. |
| `dataset` | in | `DLManagedTensor*` | The dataset. The data must be in row-major format. [dim = n_rows x n_features] |
| `centroids` | in | `DLManagedTensor*` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | in | `DLManagedTensor*` | Index of the primary cluster each row belongs to, as produced by k-means prediction. Every value must be in `[0, n_clusters)`. [len = n_rows] |
| `soar_labels` | out | `DLManagedTensor*` | Index of the secondary cluster each row is spilled to. [len = n_rows] |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
