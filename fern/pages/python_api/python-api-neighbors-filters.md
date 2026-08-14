---
slug: api-reference/python-api-neighbors-filters
---

# Filters

_Python module: `cuvs.neighbors.filters`_

## no_filter

```python
def no_filter()
```

Create a default pre-filter which filters nothing.

## from_bitmap

```python
def from_bitmap(bitmap)
```

Create a pre-filter from an array with type of uint32.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `bitmap` | `numpy.ndarray` | An array with type of `uint32` where each bit in the array corresponds to if a sample and query pair is greenlit (not filtered) or filtered. The array is row-major, meaning the bits are ordered by rows first. Each bit in a `uint32` element represents a different sample-query pair.<br /><br />- Bit value of 1: The sample-query pair is greenlit (allowed).<br />- Bit value of 0: The sample-query pair is filtered. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `filter` | `cuvs.neighbors.filters.Prefilter` | An instance of `Prefilter` that can be used to filter neighbors based on the given bitmap. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> import numpy as np
>>> from cuvs.neighbors import filters
>>>
>>> n_samples = 50000
>>> n_queries = 1000
>>>
>>> n_bitmap = np.ceil(n_samples * n_queries / 32).astype(int)
>>> bitmap = cp.random.randint(1, 100, size=(n_bitmap,), dtype=cp.uint32)
>>> prefilter = filters.from_bitmap(bitmap)
```

## from_bitset

```python
def from_bitset(bitset)
```

Create a pre-filter from an array with type of uint32.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `bitset` | `numpy.ndarray` | An array with type of `uint32` where each bit in the array corresponds to if a sample is greenlit (not filtered) or filtered. Each bit in a `uint32` element represents a different sample of the dataset.<br /><br />- Bit value of 1: The sample is greenlit (allowed).<br />- Bit value of 0: The sample pair is filtered. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `filter` | `cuvs.neighbors.filters.Prefilter` | An instance of `Prefilter` that can be used to filter neighbors based on the given bitset. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> import numpy as np
>>> from cuvs.neighbors import filters
>>>
>>> n_samples = 50000
>>> n_queries = 1000
>>>
>>> n_bitset = np.ceil(n_samples / 32).astype(int)
>>> bitset = cp.random.randint(1, 100, size=(n_bitset,), dtype=cp.uint32)
>>> prefilter = filters.from_bitset(bitset)
```

## Prefilter

```python
cdef class Prefilter
```

**Constructor**

```python
def __init__(self, cuvsFilter prefilter, parent=None)
```
