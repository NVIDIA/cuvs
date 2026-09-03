---
slug: api-reference/cpp-api-cluster-soar
---

# Soar

_Source header: `cuvs/cluster/soar.hpp`_

## SOAR hyperparameters

<a id="cluster-soar-params"></a>
### cluster::soar::params

Simple object to specify hyper-parameters for SOAR assignment.

```cpp
struct params {
  float lambda = 1.0f;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `lambda` | `float` | Weight of the projection of the secondary residual onto the primary residual in the SOAR loss. Larger values penalize secondary centroids whose residual is aligned with the primary residual, favoring complementary assignments. `0` reduces the loss to plain squared distance, which the primary centroid itself minimizes, so nothing is spilled.<br />Default: `1.0`. |

## SOAR assignment

<a id="cluster-soar-predict"></a>
### cluster::soar::predict

Assign a secondary ("spilled") cluster to each row of the dataset.

```cpp
void predict(raft::resources const& handle,
const soar::params& params,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_vector_view<const uint32_t, int64_t> labels,
raft::device_vector_view<uint32_t, int64_t> soar_labels);
```

SOAR (Spilling with Orthogonality-Amplified Residuals) picks, for each vector, a second centroid that complements the primary assignment instead of merely being the next-closest one. It minimizes the loss of Theorem 3.1 of https://arxiv.org/abs/2404.00774: for a vector `x` with primary residual `r = x - centroids[labels[i]]`,

`score(c) = ||x - c||^2 + lambda * (dot(r / ||r||, x - c))^2`

and `soar_labels[i]` is the centroid minimizing that score. Indexing a vector under both its primary and its secondary centroid improves recall for queries near a partition boundary.

Only float32 data and uint32 labels are supported.

The primary centroid is not excluded from the search, so `soar_labels[i] == labels[i]` is a possible (and meaningful) result: it says that no other centroid is worth spilling to, which is the common case for vectors in the interior of a cluster. Callers that treat SOAR as a strictly second posting list should test for this case and skip those rows.

Scratch memory scales as `n_rows * n_clusters * 4` bytes because scores against all centroids are materialized at once and are not tiled. Process the dataset in row batches to bound the peak device memory usage.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const soar::params&`](/api-reference/cpp-api-cluster-soar#cluster-soar-params) | Parameters for SOAR assignment. |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | The dataset. The data must be in row-major format. [dim = n_rows x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | in | `raft::device_vector_view<const uint32_t, int64_t>` | Index of the primary cluster each row belongs to, as produced by k-means prediction. Every value must be in `[0, n_clusters)`. [len = n_rows] |
| `soar_labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | Index of the secondary cluster each row is spilled to. [len = n_rows] |

**Returns**

`void`
