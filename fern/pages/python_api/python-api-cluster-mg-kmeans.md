---
slug: api-reference/python-api-cluster-mg-kmeans
---

# Kmeans

_Python module: `cuvs.cluster.mg.kmeans`_

## fit

`@auto_sync_multi_gpu_resources`

```python
def fit( KMeansParams params, X, centroids=None, sample_weights=None, resources=None )
```

Find clusters with single-node multi-GPU k-means using host data.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `KMeansParams` | Parameters to use to fit KMeans model. |
| `X` | `host array-like` | Training instances, shape (m, k). Must be C-contiguous float32 or float64 host data. |
| `centroids` | `host array-like, optional` | Initial centroids when ``params.init_method == "Array"`` and output centroids for all init methods. If omitted, a host NumPy output array is allocated unless ``init_method == "Array"``. |
| `sample_weights` | `host array-like, optional` | Optional weights per observation. Must be C-contiguous and have the same dtype as X. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

FitOutput
``centroids`` is a host NumPy array containing the computed centroids,
``inertia`` is the final objective value, and ``n_iter`` is the number
of iterations run.
