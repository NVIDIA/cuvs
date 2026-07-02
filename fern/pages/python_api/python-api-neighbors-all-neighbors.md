---
slug: api-reference/python-api-neighbors-all-neighbors
---

# All Neighbors

_Python module: `cuvs.neighbors.all_neighbors`_

## AllNeighborsParams

```python
cdef class AllNeighborsParams
```

Parameters for all-neighbors k-NN graph building.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `algo` | `str or cuvsAllNeighborsAlgo` | Algorithm to use for local k-NN graph building.<br />Options: "brute_force", "ivf_pq", "nn_descent" |
| `overlap_factor` | `int, default=2` | Number of clusters each point is assigned to (must be &lt; n_clusters) |
| `n_clusters` | `int, default=1` | Number of clusters/batches to partition the dataset into (&gt; overlap_factor). Use n_clusters&gt;1 to distribute the work across GPUs. |
| `metric` | `str or cuvsDistanceType, default="sqeuclidean"` | Distance metric to use for graph construction |
| `ivf_pq_params` | `cuvs.neighbors.ivf_pq.IndexParams, optional` | IVF-PQ specific parameters (used when algo="ivf_pq") |
| `nn_descent_params` | `cuvs.neighbors.nn_descent.IndexParams, optional` | NN-Descent specific parameters (used when algo="nn_descent") |

**Constructor**

```python
def __init__(self, *, algo="nn_descent", overlap_factor=2, n_clusters=1, metric="sqeuclidean", ivf_pq_params=None, nn_descent_params=None)
```

**Members**

| Name | Kind |
| --- | --- |
| `get_handle` | method |
| `algo` | property |
| `overlap_factor` | property |
| `n_clusters` | property |
| `metric` | property |

### get_handle

```python
def get_handle(self)
```

Get a pointer to the underlying C object.

### algo

```python
def algo(self)
```

Algorithm used for local k-NN graph building.

### overlap_factor

```python
def overlap_factor(self)
```

Number of clusters each point is assigned to.

### n_clusters

```python
def n_clusters(self)
```

Number of clusters/batches to partition the dataset into.

### metric

```python
def metric(self)
```

Distance metric used for graph construction.

## build

`@auto_convert_output`

```python
def build(dataset, k, params, *, indices=None, distances=None, core_distances=None, alpha=1.0, resources=None)
```

All-neighbors allows building an approximate all-neighbors knn graph.
Given a full dataset, it finds nearest neighbors for all the training
vectors in the dataset.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dataset` | `array_like` | Training dataset to build the k-NN graph for. Can be provided on host (for multi-GPU build) or device (for single-GPU build). Host vs device location is automatically detected. Supported dtype: float32 |
| `k` | `int` | Number of nearest neighbors to find for each point |
| `params` | `AllNeighborsParams` | Parameters object containing all build settings including algorithm choice and algorithm-specific parameters. |
| `indices` | `array_like, optional` | Optional output buffer for indices [num_rows x k] on device (int64). If not provided, will be allocated automatically. |
| `distances` | `array_like, optional` | Optional output buffer for distances [num_rows x k] on device (float32) |
| `core_distances` | `array_like, optional` | Optional output buffer for core distances [num_rows] on device (float32). Requires distances parameter to be provided. |
| `alpha` | `float, default=1.0` | Mutual-reachability scaling; used only when core_distances is provided |
| `resources` | `Resources or MultiGpuResources, optional` | CUDA resources to use for the operation. If not provided, a default Resources object will be created. Use MultiGpuResources to enable multi-GPU execution across multiple devices. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `indices` | `array_like` | k-NN indices for each point [num_rows x k], always on device. If indices buffer was provided, returns the same array filled with results. |
| `distances` | `array_like or None` | k-NN distances if distances buffer was provided, None otherwise |
| `core_distances` | `array_like or None` | Core distances if core_distances buffer was provided, None otherwise |
