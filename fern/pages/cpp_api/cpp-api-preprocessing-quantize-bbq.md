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

<a id="preprocessing-quantize-bbq-quantizer-view"></a>
### preprocessing::quantize::bbq::quantizer_view

Non-owning view of a BBQ quantizer, with the constness semantics of an mdspan: `quantizer_view<T>` can modify the centroid, `quantizer_view<const T>` cannot.

```cpp
template <typename T>
struct quantizer_view {
  raft::device_vector_view<T, int64_t> centroid;
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
  float centroid_norm_sq;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `centroid` | `raft::device_vector_view<T, int64_t>` | The centroid the vectors were centered around [dim]. |
| `layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) | The layout of the codes. |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | The metric the corrections were computed for. |
| `centroid_norm_sq` | `float` | The squared norm of the centroid. |

<a id="preprocessing-quantize-bbq-quantizer-view-dim"></a>
### preprocessing::quantize::bbq::quantizer_view::dim

The number of dimensions of the quantized vectors.

```cpp
[[nodiscard]] constexpr auto dim() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="preprocessing-quantize-bbq-quantizer-view-encoded-row-length"></a>
### preprocessing::quantize::bbq::quantizer_view::encoded_row_length

The bytes one encoded row occupies.

```cpp
[[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

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

<a id="preprocessing-quantize-bbq-quantizer"></a>
### preprocessing::quantize::bbq::quantizer

Better Binary Quantization ([BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch)) is a vector-quantization approach used in Elasticsearch and Apache Lucene. It builds on ideas introduced in RaBitQ([Gao and Long](https://arxiv.org/pdf/2405.12497, [Gao et al.](https://arxiv.org/pdf/2409.09913)): residual binary codes around a centroid, corrective factors, and efficient bitwise comparison of codes at different bit widths. Lucene implements this as optimized scalar quantization (OSQ) with packed and bit-plane layouts; Elasticsearch exposes it as BBQ.

BBQ in cuVS designed to be compatible with the Lucene/Elasticsearch dataset: a single shared centroid, no random rotation, and OSQ codes.

RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain additional per-vector information so search is better than naive sign-bit comparison. They differ in transformation and scale representation. RaBitQ commonly separates residual magnitude from direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector scalar intervals to interpret the compressed residual codes.

The quantizer is the manifest of the quantized dataset `cuvs::device_bbq_dataset` (see cuvs/core/dataset/bbq.hpp): the centroid the vectors were centered around, the layout of the codes, the metric the corrections were computed for, and the cached squared norm of the centroid. The encoded rows and their per-row terms are the data of the dataset.

```cpp
template <typename T>
struct quantizer {
  raft::device_vector<T, int64_t> centroid;
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
  float centroid_norm_sq;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `centroid` | `raft::device_vector<T, int64_t>` | The centroid the vectors were centered around [dim]. |
| `layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) | The layout of the codes. |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | The metric the corrections were computed for. |
| `centroid_norm_sq` | `float` | The squared norm of the centroid. |

<a id="preprocessing-quantize-bbq-quantizer-quantizer"></a>
### preprocessing::quantize::bbq::quantizer::quantizer

The quantizer owning the given centroid; its dimension is the extent of the vector.

```cpp
quantizer(raft::device_vector<T, int64_t>&& centroid_,
bbq_code_layout layout_,
cuvs::distance::DistanceType metric_,
float centroid_norm_sq_);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `centroid_` |  | `raft::device_vector<T, int64_t>&&` |  |
| `layout_` |  | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |
| `metric_` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `centroid_norm_sq_` |  | `float` |  |

**Returns**

`void`

<a id="preprocessing-quantize-bbq-quantizer-dim"></a>
### preprocessing::quantize::bbq::quantizer::dim

The number of dimensions of the quantized vectors.

```cpp
[[nodiscard]] constexpr auto dim() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="preprocessing-quantize-bbq-quantizer-encoded-row-length"></a>
### preprocessing::quantize::bbq::quantizer::encoded_row_length

The bytes one encoded row occupies.

```cpp
[[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="preprocessing-quantize-bbq-quantizer-view"></a>
### preprocessing::quantize::bbq::quantizer::view

A non-owning view of the quantizer.

```cpp
[[nodiscard]] auto view() noexcept -> view_type;
```

**Returns**

`view_type`

**Additional overload:** `preprocessing::quantize::bbq::quantizer::view`

A read-only non-owning view of the quantizer.

```cpp
[[nodiscard]] auto view() const noexcept -> const_view_type;
```

**Returns**

`const_view_type`

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
