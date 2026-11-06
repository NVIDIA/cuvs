---
slug: api-reference/cpp-api-preprocessing-quantize-scalar
---

# Scalar

_Source header: `cuvs/preprocessing/quantize/scalar.hpp`_

## Scalar quantizer utilities

<a id="preprocessing-quantize-scalar-quantization-mode"></a>
### preprocessing::quantize::scalar::quantization_mode

Select global or per-dimension quantization bounds.

```cpp
enum class quantization_mode {
  GLOBAL,
  PER_DIMENSION
};
```

**Values**

| Name | Value |
| --- | --- |
| `GLOBAL` | `` |
| `PER_DIMENSION` | `` |

<a id="preprocessing-quantize-scalar-params"></a>
### preprocessing::quantize::scalar::params

quantizer parameters.

```cpp
struct params {
  float quantile;
  quantization_mode range;
  int64_t device_buffer_samples;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `quantile` | `float` | Specifies how many outliers at top & bottom will be ignored. Needs to be within range of (0, 1]. |
| `range` | [`quantization_mode`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantization-mode) | Per-dimension training uses every row of a nonempty, finite dataset. It applies the quantile independently to each dimension to select lower and upper bounds. Transforms derive each dimension's shift from the midpoint of its bounds, rounded to the input type. Each dimension's scale fits its bounds around that shift into the symmetric range of quantized values.<br /><br />For a zero-width interval, its scale is one. Otherwise, each scale is at least the smallest positive normal value of the scale type. |
| `device_buffer_samples` | `int64_t` | Maximum rows staged on device for host training and host dataset build.<br />Default: 65536. Zero processes the entire host input in one batch. Ignored for device input. Multiple batches use two input buffers when res has an auxiliary stream pool, and one otherwise. Pageable input uses batch-sized pinned staging buffers, without registering the full input. Each input buffer uses at most device_buffer_samples * dim * sizeof(T) bytes, in addition to reduction and encoding workspace. Substitute n_rows when it is zero.<br /><br />Per-dimension host training with quantile &lt; 1 selects exact quantiles using one column of host scratch (n_rows * sizeof(T) bytes). Global training retains its sampled quantiles. |

<a id="preprocessing-quantize-scalar-quantizer-view"></a>
### preprocessing::quantize::scalar::quantizer_view

Non-owning scalar quantizer bounds, with the constness semantics of an mdspan:

quantizer_view&lt;T&gt; can modify per-dimension bounds; quantizer_view&lt;const T&gt; cannot. Global bounds and max_error are copied. The owner must outlive stream work using the vectors.

```cpp
template <typename T>
struct quantizer_view {
  std::variant<global_type, per_dim_type> value;
  scale_type max_error;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `value` | [`std::variant<global_type, per_dim_type>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-view-global-type) | Global bounds or borrowed per-dimension bounds. |
| `max_error` | `scale_type` | Maximum training-row L2 reconstruction error; see quantizer::max_error. |

<a id="preprocessing-quantize-scalar-quantizer-view-global-type"></a>
### preprocessing::quantize::scalar::quantizer_view::global_type

Global bounds. Transforms derive an affine mapping in double precision.

Encode (x - shift) / scale, rounding to nearest with ties away from zero and clamping to the full quantized range. Decode shift + scale * code, saturating to finite input limits. Equal bounds represent a constant range; every code decodes to that value.

```cpp
struct global_type {
  value_type min;
  value_type max;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `min` | `value_type` | Lower bound shared by all dimensions. |
| `max` | `value_type` | Upper bound shared by all dimensions. |

<a id="preprocessing-quantize-scalar-quantizer-view-per-dim-type"></a>
### preprocessing::quantize::scalar::quantizer_view::per_dim_type

Per-dimension bounds. Transforms derive midpoint shifts and scales from these bounds.

Encode (x - shift) / scale, rounding to nearest with ties to even and clamping to the symmetric range of quantized values. Decode shift + scale * code, saturating to finite limits of the input type.

```cpp
struct per_dim_type {
  vector_view_type mins;
  vector_view_type maxs;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `mins` | `vector_view_type` | One finite lower bound per dimension, with mins[j] &lt;= maxs[j]. |
| `maxs` | `vector_view_type` | One finite upper bound per dimension; must have the same extent as mins. |

<a id="preprocessing-quantize-scalar-quantizer-view-requires"></a>
### preprocessing::quantize::scalar::quantizer_view::requires

A mutable view decays to the read-only one, the way an mdspan does.

```cpp
template <typename U>
requires(!std::is_const_v<T> && std::is_same_v<U, const T>)
operator quantizer_view<U>() const noexcept  // NOLINT(google-explicit-constructor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `!std::is_const_v<T> && std::is_same_v<U, const T>` |  |

**Returns**

`void`

<a id="preprocessing-quantize-scalar-quantizer"></a>
### preprocessing::quantize::scalar::quantizer

Owns global bounds or device bounds for each dimension.

Move-only; copy view() to borrow the parameters.

```cpp
template <typename T>
struct quantizer {
  std::variant<global_type, per_dim_type> value;
  scale_type max_error;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `value` | [`std::variant<global_type, per_dim_type>`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer-view-global-type) | Global bounds or owning per-dimension bounds. |
| `max_error` | `scale_type` | Upper bound on the L2 error between a training row and its int8 encode/decode result. train measures every input row, including clipped outliers, and rounds the maximum up with a relative 1e-6 margin. This bound applies only to the training data, not later queries. It is infinity when constructed from parameters alone or when the error is not finite. |

<a id="preprocessing-quantize-scalar-quantizer-per-dim-type"></a>
### preprocessing::quantize::scalar::quantizer::per_dim_type

Owning per-dimension bounds.

```cpp
struct per_dim_type {
  raft::device_vector<T, int64_t> mins;
  raft::device_vector<T, int64_t> maxs;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `mins` | `raft::device_vector<T, int64_t>` |  |
| `maxs` | `raft::device_vector<T, int64_t>` |  |

<a id="preprocessing-quantize-scalar-quantizer-quantizer"></a>
### preprocessing::quantize::scalar::quantizer::quantizer

Construct default global bounds.

```cpp
quantizer();
```

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::quantizer::quantizer`

Construct from finite global bounds with min &lt;= max.

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

Copy nonempty device bounds of equal length, with finite min &lt;= max.

```cpp
quantizer(raft::resources const& res,
raft::device_vector_view<const T, int64_t> mins,
raft::device_vector_view<const T, int64_t> maxs);
```

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

Borrow mutable per-dimension bounds; the owner must outlive stream work using them.

```cpp
[[nodiscard]] auto view() noexcept -> view_type;
```

**Returns**

`view_type`

**Additional overload:** `preprocessing::quantize::scalar::quantizer::view`

Borrow read-only bounds; the owner must outlive stream work using per-dimension bounds.

```cpp
[[nodiscard]] auto view() const noexcept -> const_view_type;
```

**Returns**

`const_view_type`

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
const quantizer<double>& quantizer,
raft::device_matrix_view<const double, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<double>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::transform`

Applies global quantization to a host dataset

```cpp
void transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::host_matrix_view<const double, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Throws raft::logic_error for per-dimension parameters. Use device transforms or host build for per-dimension quantization.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<double>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer with global parameters |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

<a id="preprocessing-quantize-scalar-inverse-transform"></a>
### preprocessing::quantize::scalar::inverse_transform

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<double, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<double>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<double, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Applies inverse global quantization to a host dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<double, int64_t> out);
```

Throws raft::logic_error for per-dimension parameters. Use device inverse_transform for per-dimension quantization.

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<double>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer with global parameters |
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
const quantizer<float>& quantizer,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::transform`

Applies global quantization to a host dataset

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::host_matrix_view<const float, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Throws raft::logic_error for per-dimension parameters. Use device transforms or host build for per-dimension quantization.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer with global parameters |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<float, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<float, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Applies inverse global quantization to a host dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<float, int64_t> out);
```

Throws raft::logic_error for per-dimension parameters. Use device inverse_transform for per-dimension quantization.

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer with global parameters |
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
const quantizer<half>& quantizer,
raft::device_matrix_view<const half, int64_t> dataset,
raft::device_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<half>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<int8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::transform`

Applies global quantization to a host dataset

```cpp
void transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::host_matrix_view<const half, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Throws raft::logic_error for per-dimension parameters. Use device transforms or host build for per-dimension quantization.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<half>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer with global parameters |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<int8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::device_matrix_view<const int8_t, int64_t> dataset,
raft::device_matrix_view<half, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<half>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<half, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::scalar::inverse_transform`

Applies inverse global quantization to a host dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<half, int64_t> out);
```

Throws raft::logic_error for per-dimension parameters. Use device inverse_transform for per-dimension quantization.

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<half>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer with global parameters |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<half, int64_t>` | a row-major matrix view on host |

**Returns**

`void`
