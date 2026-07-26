---
slug: api-reference/python-api-cluster-kmeans
---

# Kmeans

_Python module: `cuvs.cluster.kmeans`_

## KMeansParams

```python
cdef class KMeansParams
```

Hyper-parameters for the kmeans algorithm

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `str` | String denoting the metric type. |
| `n_clusters` | `int` | The number of clusters to form as well as the number of centroids to generate |
| `init_method` | `str` | Method for initializing clusters. One of: "KMeansPlusPlus" : Use scalable k-means++ algorithm to select initial cluster centers "Random" : Choose 'n_clusters' observations at random from the input data "Array" : Use centroids as initial cluster centers |
| `max_iter` | `int` | Maximum number of iterations of the k-means algorithm for a single run |
| `tol` | `float` | Relative tolerance with regards to inertia to declare convergence. |
| `n_init` | `int` | Number of instance k-means algorithm will be run with different seeds |
| `oversampling_factor` | `double` | Oversampling factor for use in the k-means\|\| algorithm |
| `batch_samples` | `int` | Number of samples to process in each batch for tiled 1NN computation. Useful to optimize/control memory footprint. Default tile is [batch_samples x n_clusters]. |
| `batch_centroids` | `int` | Number of centroids to process in each batch. If 0, uses n_clusters. |
| `inertia_check` | `bool` | Deprecated and ignored. Will be removed in a future release. Inertia-based convergence checking always runs. |
| `init_size` | `int` | Number of samples to draw for KMeansPlusPlus initialization with host (out-of-core) data. When set to 0, uses the heuristic min(3 * n_clusters, n_samples). Default: 0. |
| `device_buffer_samples` | `int` | Number of samples to process per GPU batch when fitting with host (numpy) data. When set to 0, defaults to n_samples (process all at once). Only used by the batched (host-data) code path. Reducing device_buffer_samples can help reduce GPU memory pressure but increases overhead as the number of times centroid adjustments are computed increases.<br /><br />Default: 0 (process all data at once). |
| `hierarchical` | `bool` | Whether to use hierarchical (balanced) kmeans or not |
| `hierarchical_n_iters` | `int` | For hierarchical k-means , defines the number of training iterations |

**Constructor**

```python
def __init__(self, *, metric=None, n_clusters=None, init_method=None, max_iter=None, tol=None, n_init=None, oversampling_factor=None, batch_samples=None, batch_centroids=None, inertia_check=None, init_size=None, device_buffer_samples=None, hierarchical=None, hierarchical_n_iters=None)
```

**Members**

| Name | Kind |
| --- | --- |
| `metric` | property |
| `n_clusters` | property |
| `init_method` | property |
| `max_iter` | property |
| `tol` | property |
| `n_init` | property |
| `oversampling_factor` | property |
| `batch_samples` | property |
| `batch_centroids` | property |
| `init_size` | property |
| `device_buffer_samples` | property |
| `hierarchical` | property |
| `hierarchical_n_iters` | property |

### metric

```python
def metric(self)
```

### n_clusters

```python
def n_clusters(self)
```

### init_method

```python
def init_method(self)
```

### max_iter

```python
def max_iter(self)
```

### tol

```python
def tol(self)
```

### n_init

```python
def n_init(self)
```

### oversampling_factor

```python
def oversampling_factor(self)
```

### batch_samples

```python
def batch_samples(self)
```

### batch_centroids

```python
def batch_centroids(self)
```

### init_size

```python
def init_size(self)
```

### device_buffer_samples

```python
def device_buffer_samples(self)
```

### hierarchical

```python
def hierarchical(self)
```

### hierarchical_n_iters

```python
def hierarchical_n_iters(self)
```

## cluster_cost

`@auto_sync_resources`
`@auto_convert_output`

```python
def cluster_cost(X, centroids, resources=None)
```

Compute cluster cost given an input matrix and existing centroids

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `X` | `Input CUDA array interface compliant matrix shape (m, k)` |  |
| `centroids` | `Input CUDA array interface compliant matrix shape` | (n_clusters, k) |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `inertia` | `float` | The cluster cost between the input matrix and existing centroids |

**Examples**

```python
>>> import cupy as cp
>>>
>>> from cuvs.cluster.kmeans import cluster_cost
>>>
>>> n_samples = 5000
>>> n_features = 50
>>> n_clusters = 3
>>>
>>> X = cp.random.random_sample((n_samples, n_features),
...                             dtype=cp.float32)
```

```python
>>> centroids = cp.random.random_sample((n_clusters, n_features),
...                                      dtype=cp.float32)
```

```python
>>> inertia = cluster_cost(X, centroids)
```

## fit

`@auto_sync_resources`
`@auto_convert_output`

```python
def fit( KMeansParams params, X, centroids=None, sample_weights=None, resources=None )
```

Find clusters with the k-means algorithm

When X is a device array (CUDA array interface), standard on-device
k-means is used.  When X is a host array (numpy ndarray or
``__array_interface__``), data is streamed to the GPU in batches
controlled by ``params.device_buffer_samples``. For large host datasets, consider
reducing ``device_buffer_samples`` to reduce GPU memory usage.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `KMeansParams` | Parameters to use to fit KMeans model.  For host data, ``params.device_buffer_samples`` controls how many samples are sent to the GPU per batch. |
| `X` | `array-like` | Training instances, shape (m, k).  Accepts both device arrays (cupy / CUDA array interface) and host arrays (numpy). |
| `centroids` | `Optional writable CUDA array interface compliant matrix` | shape (n_clusters, k) |
| `sample_weights` | `Optional weights per observation.  Must reside on` | the same memory space as X (device or host).<br />default: None |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `centroids` | `raft.device_ndarray` | The computed centroids for each cluster |
| `inertia` | `float` | Sum of squared distances of samples to their closest cluster center |
| `n_iter` | `int` | The number of iterations used to fit the model |

**Examples**

```python
>>> import cupy as cp
>>>
>>> from cuvs.cluster.kmeans import fit, KMeansParams
>>>
>>> n_samples = 5000
>>> n_features = 50
>>> n_clusters = 3
>>>
>>> X = cp.random.random_sample((n_samples, n_features),
...                             dtype=cp.float32)
```

```python
>>> params = KMeansParams(n_clusters=n_clusters)
>>> centroids, inertia, n_iter = fit(params, X)
```

Host-data (batched) example:

```python
>>> import numpy as np
>>> X_host = np.random.random((10_000_000, 128)).astype(np.float32)
>>> params = KMeansParams(n_clusters=1000, device_buffer_samples=1_000_000)
>>> centroids, inertia, n_iter = fit(params, X_host)
```

## predict

`@auto_sync_resources`
`@auto_convert_output`

```python
def predict( KMeansParams params, X, centroids, sample_weights=None, labels=None, normalize_weight=True, resources=None )
```

Predict clusters with the k-means algorithm

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `KMeansParams` | Parameters to used in fitting KMeans model |
| `X` | `Input CUDA array interface compliant matrix shape (m, k)` |  |
| `centroids` | `CUDA array interface compliant matrix, calculated by fit` | shape (n_clusters, k) |
| `sample_weights` | `Optional input CUDA array interface compliant matrix shape` | (n_clusters, 1) default: None |
| `labels` | `Optional preallocated CUDA array interface matrix shape (m, 1)` | to hold the output |
| `normalize_weight` | `bool` | True if the weights should be normalized |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `labels` | `raft.device_ndarray` | The label for each datapoint in X |
| `inertia` | `float` | Sum of squared distances of samples to their closest cluster center |

**Examples**

```python
>>> import cupy as cp
>>>
>>> from cuvs.cluster.kmeans import fit, predict, KMeansParams
>>>
>>> n_samples = 5000
>>> n_features = 50
>>> n_clusters = 3
>>>
>>> X = cp.random.random_sample((n_samples, n_features),
...                             dtype=cp.float32)
```

```python
>>> params = KMeansParams(n_clusters=n_clusters)
>>> centroids, inertia, n_iter = fit(params, X)
>>>
>>> labels, inertia = predict(params, X, centroids)
```
