---
slug: api-reference/cpp-api-core-dataset-bbq
---

# Bbq

_Source header: `cuvs/core/dataset/bbq.hpp`_

## Types

<a id="bbq-encoding"></a>
### bbq_encoding

The data of a BBQ dataset: the rows encoded in the layout of its quantizer.

It holds arrays for an owning dataset and spans for a view of it.

```cpp
template <typename CodesT, typename VectorT, typename SumsT>
struct bbq_encoding {
  CodesT codes;
  VectorT lower_intervals;
  VectorT upper_intervals;
  VectorT additional_corrections;
  SumsT quantized_component_sums;
  VectorT dequant_delta;
  VectorT dequant_sum_delta;
  VectorT row_norm;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `codes` | `CodesT` | Encoded rows, `get_encoded_row_length(dim, layout)` bytes per row. |
| `lower_intervals` | `VectorT` | Per-row lower bound of the quantization interval, relative to the centroid. |
| `upper_intervals` | `VectorT` | Per-row upper bound of the quantization interval, relative to the centroid. |
| `additional_corrections` | `VectorT` | Per-row correction: \|\|x - c\|\|^2 for L2 metrics, x.c otherwise. |
| `quantized_component_sums` | `SumsT` | Per-row sum of the codes. |
| `dequant_delta` | `VectorT` | Per-row step between consecutive codes: (upper - lower) / (2^bits - 1). |
| `dequant_sum_delta` | `VectorT` | Per-row dequant_delta * quantized_component_sums. |
| `row_norm` | `VectorT` | Per-row squared norm in the original (un-centered) vector space, \|\|x\|\|^2. |

<a id="bbq-spec"></a>
### bbq_spec

BBQ dataset specification: device-side, as the kernels read the codes in the device memory, with

`float` per-row terms, as the kernels use. The manifest is the BBQ quantizer that produced the codes (the centroid is stored in the value type of the dataset), so the logical dimension never has to be guessed from the padded byte width of the encoded rows: the quantizer knows it.

```cpp
struct bbq_spec;
```
