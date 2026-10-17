---
slug: api-reference/cpp-api-preprocessing-quantize-bbq
---

# Bbq

_Source header: `cuvs/preprocessing/quantize/bbq.hpp`_

## Better Binary Quantization utilities

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

<a id="preprocessing-quantize-bbq-params"></a>
### preprocessing::quantize::bbq::params

```cpp
struct params {
  bbq_code_layout doc_layout;
  bbq_code_layout query_layout;
  cuvs::distance::DistanceType metric;
  float lambda;
  int iters;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `doc_layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) | The code layout of the encoded rows on the document side of a distance |
| `query_layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) | The code layout of the encoded rows on the query side of a distance. When it differs from doc_layout, the dataset is encoded a second time in this layout. |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Used for selecting the per-row additional correction: \|\|x - c\|\|^2 for L2, x.c otherwise. |
| `lambda` | `float` | Relative interval weight of the quantization error against the norm-preserving term of the loss. |
| `iters` | `int` | Coordinate descent iterations per row. |

<a id="preprocessing-quantize-bbq-get-bit-width"></a>
### preprocessing::quantize::bbq::get_bit_width

Bit width of a layout.

```cpp
constexpr auto get_bit_width(bbq_code_layout layout) noexcept -> uint32_t;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `layout` |  | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |

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
| `layout` |  | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |

**Returns**

`uint32_t`

<a id="preprocessing-quantize-bbq-quantizer"></a>
### preprocessing::quantize::bbq::quantizer

Better Binary Quantization ([BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch)) is a vector-quantization approach used in Elasticsearch and Apache Lucene. It builds on ideas introduced in RaBitQ([Gao and Long](https://arxiv.org/pdf/2405.12497, [Gao et al.](https://arxiv.org/pdf/2409.09913)): residual binary codes around a centroid, corrective factors, and efficient bitwise comparison of codes at different bit widths. Lucene implements this as optimized scalar quantization (OSQ) with packed and bit-plane layouts; Elasticsearch exposes it as BBQ.

BBQ in cuVS designed to be compatible with the Lucene/Elasticsearch dataset: a single shared centroid, no random rotation, and OSQ codes.

RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain additional per-vector information so search is better than naïve sign-bit comparison. They differ in transformation and scale representation. RaBitQ commonly separates residual magnitude from direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector scalar intervals to interpret the compressed residual codes.

```cpp
template <typename DataT>
struct quantizer {
  raft::device_vector<DataT, int64_t> centroid;
  float centroid_norm_sq;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `centroid` | `raft::device_vector<DataT, int64_t>` |  |
| `centroid_norm_sq` | `float` |  |

<a id="preprocessing-quantize-bbq-bbq-dataset-owning-storage"></a>
### preprocessing::quantize::bbq::bbq_dataset_owning_storage

A dataset encoded by a BBQ quantizer in one code layout: the codes of every row and the per-row factors needed to compare them.

The storage does not own the centroid the rows were encoded against; it reads it through `quantizer`, so the quantizer that produced it has to outlive it.

```cpp
template <typename DataT, typename IdxT>
struct bbq_dataset_owning_storage {
  raft::device_vector<float, IdxT> dequant_delta;
  raft::device_vector<float, IdxT> dequant_sum_delta;
  raft::device_vector<float, IdxT> row_norm;
  quantizer_view<DataT> quantizer;
  raft::device_matrix<uint8_t, IdxT> codes;
  raft::device_vector<float, IdxT> lower_intervals;
  raft::device_vector<float, IdxT> upper_intervals;
  raft::device_vector<float, IdxT> additional_corrections;
  raft::device_vector<int32_t, IdxT> quantized_component_sums;
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `dequant_delta` | `raft::device_vector<float, IdxT>` | Precomputed per-row dequantization factors, derived once (offline) from lower/upper_intervals and quantized_component_sums: dequant_delta = (upper-lower)/(2^bits-1) |
| `dequant_sum_delta` | `raft::device_vector<float, IdxT>` | Precomputed per-row dequantization factors, derived once (offline) from dequant_delta and quantized_component_sums: dequant_sum_delta = dequant_delta * quantized_component_sums. |
| `row_norm` | `raft::device_vector<float, IdxT>` | Squared norm of the row in original (un-centered) vector space, \|\|x\|\|^2 |
| `quantizer` | `quantizer_view<DataT>` |  |
| `codes` | `raft::device_matrix<uint8_t, IdxT>` |  |
| `lower_intervals` | `raft::device_vector<float, IdxT>` |  |
| `upper_intervals` | `raft::device_vector<float, IdxT>` |  |
| `additional_corrections` | `raft::device_vector<float, IdxT>` |  |
| `quantized_component_sums` | `raft::device_vector<int32_t, IdxT>` |  |
| `layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |

<a id="preprocessing-quantize-bbq-helpers-resolve-dequant-factors"></a>
### preprocessing::quantize::bbq::helpers::resolve_dequant_factors

Derives dequant_delta from lower/upper_intervals and the layout's code width, and

```cpp
void resolve_dequant_factors(
raft::resources const& res,
raft::device_vector_view<float, int64_t> dequant_delta,
raft::device_vector_view<float, int64_t> dequant_sum_delta,
raft::device_vector_view<const float, int64_t> lower_intervals,
raft::device_vector_view<const float, int64_t> upper_intervals,
raft::device_vector_view<const int32_t, int64_t> quantized_component_sums,
bbq_code_layout layout);
```

dequant_sum_delta from that delta and quantized_component_sums.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dequant_delta` |  | `raft::device_vector_view<float, int64_t>` |  |
| `dequant_sum_delta` |  | `raft::device_vector_view<float, int64_t>` |  |
| `lower_intervals` |  | `raft::device_vector_view<const float, int64_t>` |  |
| `upper_intervals` |  | `raft::device_vector_view<const float, int64_t>` |  |
| `quantized_component_sums` |  | `raft::device_vector_view<const int32_t, int64_t>` |  |
| `layout` |  | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |

**Returns**

`void`
