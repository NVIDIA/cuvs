---
slug: api-reference/cpp-api-core-dataset-bbq
---

# Bbq

_Source header: `cuvs/core/dataset/bbq.hpp`_

## Types

<a id="bbq-data"></a>
### bbq_data

The data of a BBQ dataset, per row: the `codes` in the layout of the manifest, together with the

per-row terms of the distance estimator

estimate = codes.shift + codes.scale * &lt;code, query&gt; + ...

where `codes.scale` is the `dequant_delta` and `codes.shift` the `additional_corrections`; then the `lower_intervals`, the `dequant_sum_delta` (the row side of the cross term with a quantized query) and the `row_norm` (the squared norm of the row, read by the cosine and the inner-product metrics). The encoder-only columns (the upper intervals and the quantized component sums) are not part of the dataset: they only feed `helpers::resolve_dequant_factors`.

It holds arrays for an owning dataset and spans for a view of it.

```cpp
template <typename CodesT, typename VectorT>
struct bbq_data {
  affine_data<CodesT, VectorT> codes;
  VectorT lower_intervals;
  VectorT dequant_sum_delta;
  VectorT row_norm;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `codes` | [`affine_data<CodesT, VectorT>`](/api-reference/cpp-api-core-dataset-affine#affine-data) |  |
| `lower_intervals` | `VectorT` |  |
| `dequant_sum_delta` | `VectorT` |  |
| `row_norm` | `VectorT` |  |

<a id="bbq-spec"></a>
### bbq_spec

BBQ dataset specification: device-side, as the kernels read the codes in the device memory, with

`float` per-row terms, as the kernels use. The manifest is the BBQ quantizer that produced the codes (the centroid is stored in the value type of the dataset), so the logical dimension never has to be guessed from the padded byte width of the encoded rows: the quantizer knows it.

```cpp
struct bbq_spec;
```
