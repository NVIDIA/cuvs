---
slug: api-reference/python-api-preprocessing-quantize-binary
---

# Binary

_Python module: `cuvs.preprocessing.quantize.binary`_

## transform

`@auto_sync_resources`
`@auto_convert_output`

```python
def transform(dataset, output=None, resources=None)
```

Applies binary quantization transform to given dataset

This applies binary quantization to a dataset, changing any positive
values to a bitwise 1. This is useful for searching with the
BitwiseHamming distance type.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dataset` | `row major host or device dataset to transform` |  |
| `output` | `optional preallocated output memory, on host or device memory` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `output` | `transformed dataset quantized into a uint8` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing.quantize import binary
>>> from cuvs.neighbors import cagra
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.standard_normal((n_samples, n_features),
...                                   dtype=cp.float32)
>>> transformed = binary.transform(dataset)
>>>
>>> # build a cagra index on the binarized data
>>> params = cagra.IndexParams(metric="bitwise_hamming",
...                            build_algo="iterative_cagra_search")
>>> idx = cagra.build(params, transformed)
```
