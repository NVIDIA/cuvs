---
slug: api-reference/cpp-api-core-dataset-bbq
---

# Bbq

_Source header: `cuvs/core/dataset/bbq.hpp`_

## bbq-dataset

<a id="preprocessing-quantize-bbq-bbq-code-layout"></a>
### preprocessing::quantize::bbq::bbq_code_layout

Storage layout of BBQ/OSQ quantized component codes in each dataset row.

```cpp
enum class bbq_code_layout {
  packed_1b,
  transposed_2b,
  transposed_4b,
  packed_4b,
  packed_7b,
  packed_8b
};
```

**Values**

| Name | Value | Description |
| --- | --- | --- |
| `packed_1b` | `` | Each dimension is quantized to a single bit and packed into bytes. Reflects Lucene's OptimizedScalarQuantizer.packAsBinary. |
| `transposed_2b` | `` | Each dimension is quantized to 2 bits, stored as 2 bitplanes. Reflects Lucene's OptimizedScalarQuantizer.transposeDibit. SIMT popc path only (paired with a transposed_4b or packed_1b operand); |
| `transposed_4b` | `` | Each dimension is quantized to 4 bits, optimized for bitwise operations. Reflects Lucene's OptimizedScalarQuantizer.transposeHalfByte. the first bit of every dimension is in the first set dimensions bits, or (dimensions/8) bytes. The second, third, and fourth bits are in the second, third, and fourth set of dimensions bits, respectively. Format used for queries. |
| `packed_4b` | `` | Each dimension is quantized to 4 bits, two values are packed into each output byte. |
| `packed_7b` | `` | Each dimension is quantized to 7 bits and treated as a signed value. |
| `packed_8b` | `` | Each dimension is quantized to 8 bits and treated as an unsigned value. |

<a id="preprocessing-quantize-bbq-get-bit-width"></a>
### preprocessing::quantize::bbq::get_bit_width

Bit width of a layout.

```cpp
constexpr auto get_bit_width(bbq_code_layout layout) noexcept -> uint32_t;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `layout` |  | [`bbq_code_layout`](/api-reference/cpp-api-core-dataset-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |

**Returns**

`uint32_t`

<a id="preprocessing-quantize-bbq-get-encoded-row-length"></a>
### preprocessing::quantize::bbq::get_encoded_row_length

Bytes one row of `dim` components occupies once encoded in `layout`.

```cpp
constexpr auto get_encoded_row_length(uint32_t dim, bbq_code_layout layout) noexcept -> uint32_t;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` |  | `uint32_t` |  |
| `layout` |  | [`bbq_code_layout`](/api-reference/cpp-api-core-dataset-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |

**Returns**

`uint32_t`

<a id="preprocessing-quantize-bbq-quantizer"></a>
### preprocessing::quantize::bbq::quantizer

Better Binary Quantization ([BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch)) is a vector-quantization approach used in Elasticsearch and Apache Lucene. It builds on ideas introduced in RaBitQ([Gao and Long](https://arxiv.org/pdf/2405.12497, [Gao et al.](https://arxiv.org/pdf/2409.09913)): residual binary codes around a centroid, corrective factors, and efficient bitwise comparison of codes at different bit widths. Lucene implements this as optimized scalar quantization (OSQ) with packed and bit-plane layouts; Elasticsearch exposes it as BBQ.

BBQ in cuVS designed to be compatible with the Lucene/Elasticsearch dataset: a single shared centroid, no random rotation, and OSQ codes.

RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain additional per-vector information so search is better than naïve sign-bit comparison. They differ in transformation and scale representation. RaBitQ commonly separates residual magnitude from direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector scalar intervals to interpret the compressed residual codes.

The quantizer is the manifest of a `cuvs::device_bbq_dataset`: what every encoding of the rows shares. The layouts and the metric belong to the encoded data.

```cpp
template <typename T>
struct quantizer {
  raft::device_vector<T, int64_t> centroid;
  raft::device_scalar<float, int64_t> centroid_norm_sq;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `centroid` | `raft::device_vector<T, int64_t>` | Centroid shared by every row; rows are encoded relative to it. |
| `centroid_norm_sq` | `raft::device_scalar<float, int64_t>` | Squared L2 norm of the centroid, \|\|c\|\|^2. |

<a id="preprocessing-quantize-bbq-quantizer-view"></a>
### preprocessing::quantize::bbq::quantizer_view

Non-owning view of a BBQ quantizer.

The view owns neither the centroid nor its norm, so the quantizer it was taken from has to outlive it.

```cpp
template <typename T>
struct quantizer_view {
  raft::device_vector_view<T, int64_t> centroid;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `centroid` | `raft::device_vector_view<T, int64_t>` | Centroid shared by every row; rows are encoded relative to it. |

<a id="preprocessing-quantize-bbq-quantizer-view-quantizer-view-u"></a>
### preprocessing::quantize::bbq::quantizer_view::quantizer_view&lt;U&gt;

A mutable view decays to the read-only one, the way an mdspan does.

```cpp
template <typename U>
operator quantizer_view<U>() const noexcept  // NOLINT(google-explicit-constructor)
requires(!std::is_const_v<T> && std::is_same_v<U, const T>);
```

**Returns**

`operator`

## Types

<a id="bbq-encoding"></a>
### bbq_encoding

One encoding of the rows of a BBQ dataset: the codes in one layout, together with the per-row

terms of the distance estimator and the encoder outputs those terms are derived from.

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
  preprocessing::quantize::bbq::bbq_code_layout layout;
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
| `layout` | [`preprocessing::quantize::bbq::bbq_code_layout`](/api-reference/cpp-api-core-dataset-bbq#preprocessing-quantize-bbq-bbq-code-layout) | Code layout of `codes`. |

<a id="bbq-data-view"></a>
### bbq_data_view

The data of a BBQ dataset view: the encoding read on the document side of a distance and the one

read on its query side. Both encode the same rows, in the same order, against the centroid of the manifest, and their corrections were computed for `metric`. A symmetric dataset refers to one encoding on both sides.

```cpp
template <typename CodesT, typename VectorT, typename SumsT>
struct bbq_data_view {
  cuvs::distance::DistanceType metric;
  encoding_type document;
  encoding_type query;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Metric the additional corrections of both encodings were computed for. |
| `document` | `encoding_type` | Encoding read on the document side of a distance. |
| `query` | `encoding_type` | Encoding read on the query side of a distance; the document encoding when symmetric. |

<a id="bbq-data"></a>
### bbq_data

The data of an owning BBQ dataset: the rows encoded in the document layout and, when the query

side of a distance reads a different layout, encoded a second time in that one.

```cpp
template <typename CodesT, typename VectorT, typename SumsT>
struct bbq_data {
  cuvs::distance::DistanceType metric;
  encoding_type document;
  std::optional<encoding_type> query;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Metric the additional corrections of both encodings were computed for. |
| `document` | `encoding_type` | Encoding read on the document side of a distance. |
| `query` | `std::optional<encoding_type>` | Encoding read on the query side of a distance; empty when the dataset is symmetric. |

<a id="bbq-spec"></a>
### bbq_spec

BBQ dataset specification: device-side, as the kernels read the codes in the device memory, with

`float` per-row terms, as the kernels use. The manifest is the BBQ quantizer that produced the codes (the centroid is stored in the value type of the dataset), so the logical dimension never has to be guessed from the padded byte width of the encoded rows: the quantizer knows it.

```cpp
struct bbq_spec;
```
