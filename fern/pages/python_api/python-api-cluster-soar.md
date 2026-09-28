---
slug: api-reference/python-api-cluster-soar
---

# Soar

_Python module: `cuvs.cluster.soar`_

## Params

```python
cdef class Params
```

Hyper-parameters for SOAR assignment.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `lambda_` | `float` | Weight of the projection of the secondary residual onto the primary residual in the SOAR loss. Larger values penalize secondary centroids whose residual is aligned with the primary residual, favoring complementary assignments. ``0`` reduces the loss to plain squared distance, which the primary centroid itself minimizes, so nothing is spilled. Named with a trailing underscore because ``lambda`` is a Python keyword (default: 1.0). |

**Constructor**

```python
def __init__(self, *, lambda_=None)
```

**Members**

| Name | Kind |
| --- | --- |
| `lambda_` | property |

### lambda_

```python
def lambda_(self)
```

## predict

`@auto_sync_resources`
`@auto_convert_output`

```python
def predict(Params params, dataset, centroids, labels, soar_labels=None, resources=None)
```

Assign a secondary ("spilled") cluster to each row of the dataset.

SOAR (Spilling with Orthogonality-Amplified Residuals) picks, for each
vector, a second centroid that complements the primary assignment instead
of merely being the next-closest one. It minimizes the loss of Theorem 3.1
of https://arxiv.org/abs/2404.00774: for a vector ``x`` with primary
residual ``r = x - centroids[labels[i]]``,

``score(c) = \|\|x - c\|\|^2 + lambda * (dot(r / \|\|r\|\|, x - c))^2``

and ``soar_labels[i]`` is the centroid minimizing that score. Indexing a
vector under both its primary and its secondary centroid improves recall
for queries near a partition boundary.

The primary centroid is not excluded from the search, so
``soar_labels[i] == labels[i]`` is a possible (and meaningful) result: it
says that no other centroid is worth spilling to, which is the common case
for vectors in the interior of a cluster. Callers that treat SOAR as a
strictly second posting list should test for this case and skip those rows.

Scratch memory scales as ``n_rows * n_clusters * 4`` bytes because scores
against all centroids are materialized at once and are not tiled. Process
the dataset in row batches to bound the peak device memory usage.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `Params` | Parameters for SOAR assignment. |
| `dataset` | `CUDA array interface compliant matrix, row major, float32` | shape (n_rows, n_features) |
| `centroids` | `CUDA array interface compliant matrix, row major, float32` | Cluster centroids, shape (n_clusters, n_features) |
| `labels` | `CUDA array interface compliant vector, uint32 or int32` | Index of the primary cluster each row belongs to, as produced by k-means prediction. Every value must be in ``[0, n_clusters)``. shape (n_rows,) |
| `soar_labels` | `Optional preallocated CUDA array interface vector to hold` | the output, shape (n_rows,). Must have the same dtype as ``labels``. When None, an array matching the dtype of ``labels`` is allocated. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `soar_labels` | `raft.device_ndarray` | Index of the secondary cluster each row is spilled to. |

**Examples**

```python
>>> import cupy as cp
>>>
>>> from cuvs.cluster.kmeans import KMeansParams
>>> from cuvs.cluster.kmeans import fit as kmeans_fit
>>> from cuvs.cluster.kmeans import predict as kmeans_predict
>>> from cuvs.cluster.soar import Params, predict
>>>
>>> n_samples = 5000
>>> n_features = 50
>>> n_clusters = 50
>>>
>>> X = cp.random.random_sample((n_samples, n_features),
...                             dtype=cp.float32)
>>>
>>> # SOAR needs centroids and a primary label per row, so k-means runs
>>> # first
>>> kmeans_params = KMeansParams(n_clusters=n_clusters)
>>> centroids, inertia, n_iter = kmeans_fit(kmeans_params, X)
>>> labels, inertia = kmeans_predict(kmeans_params, X, centroids)
>>>
>>> soar_labels = predict(Params(), X, centroids, labels)
```
