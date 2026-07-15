---
slug: api-reference/python-api-preprocessing-quantize-scalar
---

# Scalar

_Python module: `cuvs.preprocessing.quantize.scalar`_

## Quantizer

```python
cdef class Quantizer
```

Defines and stores scalar for quantisation upon training

The quantization is performed by a linear mapping of an interval in the
float data type to the full range of the quantized int type.

**Members**

| Name | Kind |
| --- | --- |
| `min` | property |
| `max` | property |

### min

```python
def min(self)
```

### max

```python
def max(self)
```

## QuantizerParams

```python
cdef class QuantizerParams
```

Parameters for scalar quantization

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `quantile` | `float` | specifies how many outliers at top & bottom will be ignored needs to be within range of (0, 1] |

**Constructor**

```python
def __init__(self, *, quantile=None)
```

**Members**

| Name | Kind |
| --- | --- |
| `quantile` | property |

### quantile

```python
def quantile(self)
```

## inverse_transform

`@auto_sync_resources`
`@auto_convert_output`

```python
def inverse_transform(Quantizer quantizer, dataset, output=None, resources=None)
```

Perform inverse quantization step on previously quantized dataset

Note that depending on the chosen data types train dataset the conversion
is not lossless.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `quantizer` | `trained Quantizer object` |  |
| `dataset` | `row major host or device dataset to transform` |  |
| `output` | `optional preallocated output memory, on host or device` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `output` | `transformed dataset with scalar quantization reversed` |  |

## train

`@auto_sync_resources`

```python
def train(QuantizerParams params, dataset, resources=None)
```

Initializes a scalar quantizer to be used later for quantizing the dataset.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `QuantizerParams object` |  |
| `dataset` | `row major host or device dataset` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `quantizer` | `cuvs.preprocessing.quantize.scalar.Quantizer` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing.quantize import scalar
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> params = scalar.QuantizerParams(quantile=0.99)
>>> quantizer = scalar.train(params, dataset)
>>> transformed = scalar.transform(quantizer, dataset)
```

## transform

`@auto_sync_resources`
`@auto_convert_output`

```python
def transform(Quantizer quantizer, dataset, output=None, resources=None)
```

Applies quantization transform to given dataset

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `quantizer` | `trained Quantizer object` |  |
| `dataset` | `row major host or device dataset to transform` |  |
| `output` | `optional preallocated output memory, on host or device memory` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `output` | `transformed dataset quantized into a int8` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing.quantize import scalar
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> params = scalar.QuantizerParams(quantile=0.99)
>>> quantizer = scalar.train(params, dataset)
>>> transformed = scalar.transform(quantizer, dataset)
```
