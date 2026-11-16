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

Non-owning view of a BBQ quantizer.

The view owns neither the centroid nor its norm, so the quantizer it was taken from has to outlive it.

```cpp
template <typename T>
struct quantizer_view {
  raft::device_vector_view<T, int64_t> centroid;
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `centroid` | `raft::device_vector_view<T, int64_t>` | Centroid shared by every row; rows are encoded relative to it. |
| `layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) | Code layout of the encoded rows. |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Metric the per-row additional corrections are computed for. |

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

RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain additional per-vector information so search is better than naïve sign-bit comparison. They differ in transformation and scale representation. RaBitQ commonly separates residual magnitude from direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector scalar intervals to interpret the compressed residual codes.

The quantizer is the manifest of a `cuvs::device_bbq_dataset`: it states how the rows of the dataset are encoded. Two datasets encoding the same rows in different layouts against the same centroid form the document and query sides of an asymmetric distance.

```cpp
template <typename T>
struct quantizer {
  raft::device_vector<T, int64_t> centroid;
  raft::device_scalar<float, int64_t> centroid_norm_sq;
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `centroid` | `raft::device_vector<T, int64_t>` | Centroid shared by every row; rows are encoded relative to it. |
| `centroid_norm_sq` | `raft::device_scalar<float, int64_t>` | Squared L2 norm of the centroid, \|\|c\|\|^2. |
| `layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) | Code layout of the encoded rows. |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Metric the per-row additional corrections are computed for. |

<a id="preprocessing-quantize-bbq-params"></a>
### preprocessing::quantize::bbq::params

```cpp
struct params {
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
  float lambda;
  int iters;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) | The code layout of the encoded rows. |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Used for selecting the per-row additional correction: \|\|x - c\|\|^2 for L2, x.c otherwise. |
| `lambda` | `float` | Relative interval weight of the quantization error against the norm-preserving term of the loss. |
| `iters` | `int` | Coordinate descent iterations per row. |

<a id="preprocessing-quantize-bbq-build"></a>
### preprocessing::quantize::bbq::build

Builds a BBQ quantizer from a dataset: computes the centroid shared by every row, and its squared norm, into `quantizer`.

```cpp
void build(raft::resources const& res,
raft::device_matrix_view<const float, int64_t> dense_dataset,
quantizer_view<float> quantizer);
```

Its dimension has to match the one of `dense_dataset`.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `dense_dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `quantizer` | out | [`quantizer_view<float>`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-quantizer-view) | a view of the BBQ quantizer to write the centroid into |

**Returns**

`void`

<a id="preprocessing-quantize-bbq-make-device-bbq-dataset"></a>
### preprocessing::quantize::bbq::make_device_bbq_dataset

Allocates a BBQ dataset of the shape `transform` writes, leaving its contents uninitialized.

```cpp
cuvs::dataset<float, int64_t, cuvs::bbq_spec> make_device_bbq_dataset(raft::resources const& res,
const params params,
int64_t n_rows,
uint32_t dim);
```

The rows are allocated in `params.layout`. The quantizer records `params.layout` and `params.metric`, and holds room for a centroid of `dim` components.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-params) | configuration for BBQ encoding; only layout and metric are used |
| `n_rows` | in | `int64_t` | number of rows |
| `dim` | in | `uint32_t` | number of components of each row before encoding |

**Returns**

[`cuvs::dataset<float, int64_t, cuvs::bbq_spec>`](/api-reference/cpp-api-core-dataset-bbq#bbq-spec)

<a id="preprocessing-quantize-bbq-transform"></a>
### preprocessing::quantize::bbq::transform

Encodes a dataset into a BBQ dataset, against the centroid in the manifest of the BBQ dataset.

```cpp
void transform(raft::resources const& res,
const params params,
raft::device_matrix_view<const float, int64_t> dense_dataset,
cuvs::dataset_view<float, int64_t, cuvs::bbq_spec> out);
```

hold the quantizer already, from a previous call to `build`, and its layout and metric have to be the ones of `params`.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-params) | configuration for BBQ encoding, e.g. layout, metric, lambda, iters |
| `dense_dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `out` | inout | [`cuvs::dataset_view<float, int64_t, cuvs::bbq_spec>`](/api-reference/cpp-api-core-dataset-view#dataset-view) | a view of the BBQ dataset whose manifest holds the quantizer, to write the encoding into; a `cuvs::device_bbq_dataset<float, int64_t>::view_type` |

**Returns**

`void`
