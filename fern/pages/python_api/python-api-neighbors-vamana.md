---
slug: api-reference/python-api-neighbors-vamana
---

# Vamana

_Python module: `cuvs.neighbors.vamana`_

## Index

```python
cdef class Index
```

Vamana index object. This object stores the trained Vamana index state
which can be used to perform nearest neighbors searches.

**Members**

| Name | Kind |
| --- | --- |
| `trained` | property |

### trained

```python
def trained(self)
```

## IndexParams

```python
cdef class IndexParams
```

Parameters for building a Vamana index

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `str, default="sqeuclidean"` | String denoting the metric type. Supported metrics include:<br />- "sqeuclidean"<br />- "l2" |
| `graph_degree` | `int, default=32` | Maximum degree of graph; corresponds to the R parameter of Vamana algorithm in the literature. |
| `visited_size` | `int, default=64` | Maximum number of visited nodes per search during Vamana algorithm. Loosely corresponds to the L parameter in the literature. |
| `vamana_iters` | `float, default=1` | Number of Vamana vector insertion iterations (each iteration inserts all vectors). |
| `alpha` | `float, default=1.2` | Alpha for pruning parameter. Used to determine how aggressive the pruning will be. |
| `max_fraction` | `float, default=0.06` | Maximum fraction of dataset inserted per batch. Larger max batch decreases graph quality, but improves speed. |
| `batch_base` | `float, default=2.0` | Base of growth rate of batch sizes. |
| `queue_size` | `int, default=127` | Size of candidate queue structure - should be (2^x)-1. |
| `reverse_batchsize` | `int, default=1000000` | Max batchsize of reverse edge processing (reduces memory footprint). |

**Constructor**

```python
def __init__(self, *, metric="sqeuclidean", graph_degree=32, visited_size=64, vamana_iters=1, alpha=1.2, max_fraction=0.06, batch_base=2.0, queue_size=127, reverse_batchsize=1000000)
```

**Members**

| Name | Kind |
| --- | --- |
| `metric` | property |
| `graph_degree` | property |
| `visited_size` | property |
| `vamana_iters` | property |
| `alpha` | property |
| `max_fraction` | property |
| `batch_base` | property |
| `queue_size` | property |
| `reverse_batchsize` | property |

### metric

```python
def metric(self)
```

### graph_degree

```python
def graph_degree(self)
```

### visited_size

```python
def visited_size(self)
```

### vamana_iters

```python
def vamana_iters(self)
```

### alpha

```python
def alpha(self)
```

### max_fraction

```python
def max_fraction(self)
```

### batch_base

```python
def batch_base(self)
```

### queue_size

```python
def queue_size(self)
```

### reverse_batchsize

```python
def reverse_batchsize(self)
```

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, resources=None)
```

Build the Vamana index from the dataset for efficient search.

The build utilities the Vamana insertion-based algorithm to create
the graph. The algorithm starts with an empty graph and iteratively
inserts batches of nodes. Each batch involves performing a greedy
search for each vector to be inserted, and inserting it with edges to
all nodes traversed during the search. Reverse edges are also inserted
and robustPrune is applied to improve graph quality. The index_params
struct controls the degree of the final graph.

The following distance metrics are supported:
- L2Expanded

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `IndexParams object` |  |
| `dataset` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float, int8, uint8] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.vamana.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import vamana
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> build_params = vamana.IndexParams(metric="sqeuclidean")
>>> index = vamana.build(build_params, dataset)
>>> # Serialize index to file for later use with CPU DiskANN
>>> vamana.save("my_index.bin", index)
```

## save

`@auto_sync_resources`

```python
def save(filename, Index index, bool include_dataset=True, resources=None)
```

Saves the index to a file.

Matches the file format used by the DiskANN open-source repository,
allowing cross-compatibility.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `index` | `Index` | Trained Vamana index. |
| `include_dataset` | `bool` | Whether or not to write out the dataset along with the index. Including the dataset in the serialized index will use extra disk space, and might not be desired if you already have a copy of the dataset on disk. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import vamana
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = vamana.build(vamana.IndexParams(), dataset)
>>> # Serialize and save the vamana index
>>> vamana.save("my_index.bin", index)
```
