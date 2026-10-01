---
slug: api-reference/python-api-neighbors-nn-descent
---

# NN Descent

_Python module: `cuvs.neighbors.nn_descent`_

## Index

```python
cdef class Index
```

NN-Descent index object. This object stores the trained NN-Descent index,
which can be used to get the NN-Descent graph and distances after
building

**Members**

| Name | Kind |
| --- | --- |
| `trained` | property |
| `graph` | property |
| `distances` | property |

### trained

```python
def trained(self)
```

### graph

```python
def graph(self)
```

### distances

```python
def distances(self)
```

## IndexParams

```python
cdef class IndexParams
```

Parameters to build NN-Descent Index

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `str, default = "sqeuclidean"` | String denoting the metric type. Supported metrics are `l2`, `euclidean`, `sqeuclidean`, `inner_product`, `cosine`, and `bitwise_hamming` (`bitwise_hamming` is for int8 and uint8 data types only) |
| `graph_degree` | `int` | For an input dataset of dimensions (N, D), determines the final dimensions of the all-neighbors knn graph which turns out to be of dimensions (N, graph_degree) |
| `intermediate_graph_degree` | `int` | Internally, nn-descent builds an all-neighbors knn graph of dimensions (N, intermediate_graph_degree) before selecting the final `graph_degree` neighbors. It's recommended that `intermediate_graph_degree` &gt;= 1.5 * graph_degree |
| `max_iterations` | `int` | The number of iterations that nn-descent will refine the graph for. More iterations produce a better quality graph at cost of performance |
| `termination_threshold` | `float` | The delta at which nn-descent will terminate its iterations |
| `return_distances` | `bool` | Whether to return distances array |
| `dist_comp_dtype` | `str, default = "auto"` | Dtype to use for distance computation. Supported dtypes are `auto`, `fp32`, and `fp16` `auto` automatically determines the best dtype for distance computation based on the dataset dimensions. `fp32` uses fp32 distance computation for better precision at the cost of performance and memory usage. This option is only valid when data type is fp32. `fp16` uses fp16 distance computation for better performance and memory usage at the cost of precision. |

**Constructor**

```python
def __init__(self, *, metric=None, metric_arg=None, graph_degree=None, intermediate_graph_degree=None, max_iterations=None, termination_threshold=None, return_distances=None, dist_comp_dtype="auto" )
```

**Members**

| Name | Kind |
| --- | --- |
| `metric` | property |
| `metric_arg` | property |
| `graph_degree` | property |
| `intermediate_graph_degree` | property |
| `max_iterations` | property |
| `termination_threshold` | property |
| `get_handle` | method |

### metric

```python
def metric(self)
```

### metric_arg

```python
def metric_arg(self)
```

### graph_degree

```python
def graph_degree(self)
```

### intermediate_graph_degree

```python
def intermediate_graph_degree(self)
```

### max_iterations

```python
def max_iterations(self)
```

### termination_threshold

```python
def termination_threshold(self)
```

### get_handle

```python
def get_handle(self)
```

Get a pointer to the underlying C object.

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, graph=None, resources=None)
```

Build KNN graph from the dataset

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `cuvs.neighbors.nn_descent.IndexParams` |  |
| `dataset` | `Array interface compliant matrix, on either host or device memory` | Supported dtype [float, int8, uint8] |
| `graph` | `Optional host matrix for storing output graph` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.nn_descent.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import nn_descent
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> build_params = nn_descent.IndexParams(metric="sqeuclidean")
>>> index = nn_descent.build(build_params, dataset)
>>> graph = index.graph
```
