---
slug: api-reference/cpp-api-preprocessing-quantize-scalar
---

# Scalar

_Source header: `cuvs/preprocessing/quantize/scalar.hpp`_

## Scalar quantizer utilities

<a id="preprocessing-quantize-scalar-range-mode"></a>
### preprocessing::quantize::scalar::range_mode

Select a global interval or per-dimension shifts with a shared scale.

```cpp
enum class range_mode {
  global,
  per_dimension
};
```

**Values**

| Name | Value |
| --- | --- |
| `global` | `` |
| `per_dimension` | `` |

<a id="preprocessing-quantize-scalar-params"></a>
### preprocessing::quantize::scalar::params

quantizer parameters.

```cpp
struct params {
  float quantile;
  range_mode range;
  int64_t device_buffer_samples;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `quantile` | `float` | Specifies how many outliers at top & bottom will be ignored. Needs to be within range of (0, 1]. |
| `range` | [`range_mode`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-range-mode) | Per-dimension training uses every row of a nonempty, finite dataset. It applies the quantile independently to each dimension to select lower and upper bounds. Each dimension's shift is the midpoint of its bounds, rounded to the input type. A shared scale fits every dimension's bounds around its stored shift into the symmetric range of quantized values.<br /><br />If every interval has zero width, the scale is one. Otherwise, the scale is at least the smallest positive normal value of the scale type. Training synchronizes the stream before returning the scale to the host. |
| `device_buffer_samples` | `int64_t` | Maximum rows staged on device for host per-dimension training and host dataset build.<br />Default: 65536. Zero processes the entire host input in one batch. Ignored for device input. Multiple batches use two input buffers when res has an auxiliary stream pool, and one otherwise. Pageable input uses batch-sized pinned staging buffers, without registering the full input. Each input buffer uses at most device_buffer_samples * dim * sizeof(T) bytes, in addition to reduction and encoding workspace. Substitute n_rows when it is zero.<br /><br />Per-dimension host training with quantile &lt; 1 selects exact quantiles using one column of host scratch (n_rows * sizeof(T) bytes). Global training retains its sampled quantiles. |

<a id="preprocessing-quantize-scalar-quantizer-const-view"></a>
### preprocessing::quantize::scalar::quantizer_const_view

Read-only scalar quantizer parameters. The owner must outlive stream work using shifts.

```cpp
template <typename T>
struct quantizer_const_view {
  std::variant<range_type, shifts_type> value;
  scale_type max_error;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `value` | [`std::variant<range_type, shifts_type>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view-range-type) | Global interval or borrowed per-dimension shifts and scale. |
| `max_error` | `scale_type` | Maximum training-row L2 reconstruction error; see quantizer::max_error. |

<a id="preprocessing-quantize-scalar-quantizer-const-view-range-type"></a>
### preprocessing::quantize::scalar::quantizer_const_view::range_type

A global interval mapped to the full range of the quantized type.

```cpp
struct range_type {
  T min_;
  T max_;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `min_` | `T` |  |
| `max_` | `T` |  |

<a id="preprocessing-quantize-scalar-quantizer-const-view-shifts-type"></a>
### preprocessing::quantize::scalar::quantizer_const_view::shifts_type

Encode (x - shifts[j]) / scale, rounding to nearest with ties to even and

clamping to the symmetric range of quantized values. Decode shifts[j] + scale * code, saturating to finite limits of the input type.

```cpp
struct shifts_type {
  raft::device_vector_view<const T, int64_t> shifts;
  scale_type scale;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `shifts` | `raft::device_vector_view<const T, int64_t>` |  |
| `scale` | `scale_type` |  |

<a id="preprocessing-quantize-scalar-quantizer"></a>
### preprocessing::quantize::scalar::quantizer

Owns either a global interval or device shifts and a shared step size.

Move-only; copy view() to borrow the parameters. Migration: replace q.min_ with std::get&lt;quantizer&lt;T&gt;::range_type&gt;(q.value).min_ (likewise for max_).

```cpp
template <typename T>
struct quantizer {
  std::variant<range_type, shifts_type> value;
  scale_type max_error;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `value` | [`std::variant<range_type, shifts_type>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view-range-type) | Global interval or owning per-dimension shifts and scale. |
| `max_error` | `scale_type` | Upper bound on the L2 error between a training row and its int8 encode/decode result. train measures every input row, including clipped outliers, and rounds the maximum up with a relative 1e-6 margin. This bound applies only to the training data, not later queries. It is infinity when constructed from parameters alone or when the error is not finite. |

<a id="preprocessing-quantize-scalar-quantizer-shifts-type"></a>
### preprocessing::quantize::scalar::quantizer::shifts_type

Owning per-dimension parameters.

```cpp
struct shifts_type {
  raft::device_vector<T, int64_t> shifts;
  scale_type scale;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `shifts` | `raft::device_vector<T, int64_t>` |  |
| `scale` | `scale_type` |  |

<a id="preprocessing-quantize-scalar-quantizer-quantizer"></a>
### preprocessing::quantize::scalar::quantizer::quantizer

Construct an empty global interval.

```cpp
quantizer();
```

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::quantizer::quantizer`

Construct a global interval.

```cpp
quantizer(T min, T max);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `min` |  | `T` |  |
| `max` |  | `T` |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::quantizer::quantizer`

Construct from nonempty device bounds of equal length, with finite min &lt;= max.

```cpp
quantizer(raft::resources const& res,
raft::device_vector_view<const T, int64_t> mins,
raft::device_vector_view<const T, int64_t> maxs);
```

Derives midpoint shifts and a shared scale; synchronizes before returning.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `mins` |  | `raft::device_vector_view<const T, int64_t>` |  |
| `maxs` |  | `raft::device_vector_view<const T, int64_t>` |  |

**Returns**

`void`

<a id="preprocessing-quantize-scalar-quantizer-view"></a>
### preprocessing::quantize::scalar::quantizer::view

Borrow read-only parameters; the owner must outlive stream work using shifts.

```cpp
[[nodiscard]] auto view() const noexcept -> view_type;
```

**Returns**

`view_type`

<a id="preprocessing-quantize-scalar-quantizer-view-type"></a>
### preprocessing::quantize::scalar::quantizer::view_type

An owning quantizer can be passed wherever a read-only view is accepted.

```cpp
operator view_type() const noexcept;
```

**Returns**

`operator`

<a id="preprocessing-quantize-scalar-train"></a>
### preprocessing::quantize::scalar::train

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<double> train(raft::resources const& res,
const params params,
raft::device_matrix_view<const double, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-params) | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |

**Returns**

[`quantizer<double>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer)

**Additional overload:** `preprocessing::quantize::scalar::train`

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<double> train(raft::resources const& res,
const params params,
raft::host_matrix_view<const double, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-params) | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |

**Returns**

[`quantizer<double>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer)

<a id="preprocessing-quantize-scalar-transform"></a>
### preprocessing::quantize::scalar::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
quantizer_const_view<double> quantizer,
raft::device_matrix_view<const double, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<double>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::transform`

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
quantizer_const_view<double> quantizer,
raft::host_matrix_view<const double, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<double>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

<a id="preprocessing-quantize-scalar-inverse-transform"></a>
### preprocessing::quantize::scalar::inverse_transform

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
quantizer_const_view<double> quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<double, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<double>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<double, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
quantizer_const_view<double> quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<double, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<double>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<double, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::train`

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<float> train(raft::resources const& res,
const params params,
raft::device_matrix_view<const float, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-params) | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |

**Returns**

[`quantizer<float>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer)

**Additional overload:** `preprocessing::quantize::scalar::train`

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<float> train(raft::resources const& res,
const params params,
raft::host_matrix_view<const float, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-params) | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |

**Returns**

[`quantizer<float>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer)

**Additional overload:** `preprocessing::quantize::scalar::transform`

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
quantizer_const_view<float> quantizer,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<float>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::transform`

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
quantizer_const_view<float> quantizer,
raft::host_matrix_view<const float, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<float>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
quantizer_const_view<float> quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<float, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<float>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<float, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
quantizer_const_view<float> quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<float, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<float>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<float, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::train`

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<half> train(raft::resources const& res,
const params params,
raft::device_matrix_view<const half, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-params) | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |

**Returns**

[`quantizer<half>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer)

**Additional overload:** `preprocessing::quantize::scalar::train`

Initializes a scalar quantizer to be used later for quantizing the dataset.

```cpp
quantizer<half> train(raft::resources const& res,
const params params,
raft::host_matrix_view<const half, int64_t> dataset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-params) | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |

**Returns**

[`quantizer<half>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer)

**Additional overload:** `preprocessing::quantize::scalar::transform`

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
quantizer_const_view<half> quantizer,
raft::device_matrix_view<const half, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<half>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::transform`

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
quantizer_const_view<half> quantizer,
raft::host_matrix_view<const half, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<half>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
quantizer_const_view<half> quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<half, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<half>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<half, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
quantizer_const_view<half> quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<half, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`quantizer_const_view<half>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-const-view) | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<half, int64_t>` | a row-major matrix view on host |

**Returns**

`void`
