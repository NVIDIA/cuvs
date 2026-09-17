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

Multiple host batches are automatically double-buffered on each GPU.
Configure ``resources.set_stream_pool(1)`` for transfer/compute overlap;
without it, execution is correct but serialized. Pinned host memory is
crucial for performance: pageable or unregistered memory-mapped input
degrades throughput rapidly.

A per-device memory pool is optional but recommended. Configure both pools
before calling ``fit``.

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

**Examples**

```python
>>> import numpy as np
>>> import cupyx
>>> from cuvs.cluster.kmeans import KMeansParams
>>> from cuvs.cluster.mg import kmeans
>>> from cuvs.common import MultiGpuResources
>>> X = cupyx.empty_pinned((10_000_000, 128), dtype=np.float32)
>>> _ = np.random.default_rng().random(
...     X.shape, dtype=np.float32, out=X)
>>> params = KMeansParams(n_clusters=1000,
...                       device_buffer_samples=1_000_000)
>>> resources = MultiGpuResources()
>>> resources.set_memory_pool(80)
>>> resources.set_stream_pool(1)
>>> result = kmeans.fit(params, X, resources=resources)
>>> resources.sync()
```
