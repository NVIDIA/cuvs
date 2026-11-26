---
slug: api-reference/python-api-preprocessing-quantize-pq
---

# PQ

_Python module: `cuvs.preprocessing.quantize.pq`_

## Quantizer

```python
cdef class Quantizer
```

Defines and stores Product Quantizer upon training

The quantization is performed by a linear mapping of an interval in the
float data type to the full range of the quantized int type.

**Members**

| Name | Kind |
| --- | --- |
| `pq_bits` | property |
| `pq_dim` | property |
| `pq_codebook` | property |
| `vq_codebook` | property |
| `encoded_dim` | property |
| `use_vq` | property |

### pq_bits

```python
def pq_bits(self)
```

### pq_dim

```python
def pq_dim(self)
```

### pq_codebook

```python
def pq_codebook(self)
```

Returns the PQ codebook

### vq_codebook

```python
def vq_codebook(self)
```

Returns the VQ codebook

### encoded_dim

```python
def encoded_dim(self)
```

Returns the encoded dimension of the quantized dataset

### use_vq

```python
def use_vq(self)
```

## QuantizerParams

```python
cdef class QuantizerParams
```

Parameters for product quantization

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `int` | specifies the bit length of the vector element after compression by PQ<br />possible values: within [4, 16] |
| `pq_dim` | `int` | specifies the dimensionality of the vector after compression by PQ |
| `use_subspaces` | `bool` | specifies whether to use subspaces for product quantization (PQ). When true, one PQ codebook is used for each subspace. Otherwise, a single PQ codebook is used. |
| `use_vq` | `bool` | specifies whether to use Vector Quantization (KMeans) before product quantization (PQ). |
| `vq_n_centers` | `int` | specifies the number of centers for the vector quantizer. When zero, an optimal value is selected using a heuristic. When one, only product quantization is used. |
| `kmeans_n_iters` | `int` | specifies the number of iterations searching for kmeans centers |
| `pq_kmeans_type` | `str` | specifies the type of kmeans algorithm to use for PQ training<br />possible values: "kmeans", "kmeans_balanced" |
| `max_train_points_per_pq_code` | `int` | specifies the max number of data points to use per PQ code during PQ codebook training. Using more data points per PQ code may increase the quality of PQ codebook but may also increase the build time. |
| `max_train_points_per_vq_cluster` | `int` | specifies the max number of data points to use per VQ cluster. |

**Constructor**

```python
def __init__(self, *, pq_bits=8, pq_dim=0, use_subspaces=True, use_vq=False, vq_n_centers=0, kmeans_n_iters=25, pq_kmeans_type="kmeans_balanced", max_train_points_per_pq_code=256, max_train_points_per_vq_cluster=1024)
```

**Members**

| Name | Kind |
| --- | --- |
| `pq_bits` | property |
| `pq_dim` | property |
| `vq_n_centers` | property |
| `kmeans_n_iters` | property |
| `pq_kmeans_type` | property |
| `max_train_points_per_pq_code` | property |
| `max_train_points_per_vq_cluster` | property |
| `use_vq` | property |
| `use_subspaces` | property |

### pq_bits

```python
def pq_bits(self)
```

### pq_dim

```python
def pq_dim(self)
```

### vq_n_centers

```python
def vq_n_centers(self)
```

### kmeans_n_iters

```python
def kmeans_n_iters(self)
```

### pq_kmeans_type

```python
def pq_kmeans_type(self)
```

### max_train_points_per_pq_code

```python
def max_train_points_per_pq_code(self)
```

### max_train_points_per_vq_cluster

```python
def max_train_points_per_vq_cluster(self)
```

### use_vq

```python
def use_vq(self)
```

### use_subspaces

```python
def use_subspaces(self)
```

## build

`@auto_sync_resources`

```python
def build(QuantizerParams params, dataset, resources=None)
```

Builds a Product Quantizer to be used later for quantizing
the dataset.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `QuantizerParams object` |  |
| `dataset` | `row major dataset on host or device memory. FP32` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `quantizer` | `cuvs.preprocessing.quantize.pq.Quantizer` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing.quantize import pq
>>> n_samples = 5000
>>> n_features = 64
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16)
>>> quantizer = pq.build(params, dataset)
>>> transformed, _ = pq.transform(quantizer, dataset)
```

## transform

`@auto_sync_resources`
`@auto_convert_output`

```python
def transform(Quantizer quantizer, dataset, codes_output=None, vq_labels=None, resources=None)
```

Applies Product Quantization transform to given dataset

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `quantizer` | `trained Quantizer object` |  |
| `dataset` | `row major dataset on host or device memory. FP32` |  |
| `codes_output` | `optional preallocated output memory, on device memory` |  |
| `vq_labels` | `optional preallocated output memory for VQ labels, on device memory` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `codes_output` | `transformed dataset quantized into a uint8` |  |
| `vq_labels` | `VQ labels when VQ is used, None otherwise` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing.quantize import pq
>>> n_samples = 5000
>>> n_features = 64
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16)
>>> quantizer = pq.build(params, dataset)
>>> transformed, _ = pq.transform(quantizer, dataset)
```

## inverse_transform

`@auto_sync_resources`
`@auto_convert_output`

```python
def inverse_transform(Quantizer quantizer, codes, output=None, vq_labels=None, resources=None)
```

Applies Product Quantization inverse transform to given codes

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `quantizer` | `trained Quantizer object` |  |
| `codes` | `row major device codes to inverse transform. uint8` |  |
| `output` | `optional preallocated output memory, on device memory` |  |
| `vq_labels` | `optional VQ labels when VQ is used, on device memory` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `output` | `Original dataset reconstructed from quantized codes` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing.quantize import pq
>>> n_samples = 5000
>>> n_features = 64
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16, use_vq=True)
>>> quantizer = pq.build(params, dataset)
>>> transformed, vq_labels = pq.transform(quantizer, dataset)
>>> reconstructed = pq.inverse_transform(quantizer, transformed, vq_labels=vq_labels)
```
