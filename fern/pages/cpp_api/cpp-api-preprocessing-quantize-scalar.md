---
slug: api-reference/cpp-api-preprocessing-quantize-scalar
---

# Scalar

_Source header: `cuvs/preprocessing/quantize/scalar.hpp`_

## Scalar quantizer utilities

<a id="preprocessing-quantize-scalar-params"></a>
### preprocessing::quantize::scalar::params

quantizer parameters.

```cpp
struct params {
  float quantile;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `quantile` | `float` | Specifies how many outliers at top & bottom will be ignored. Needs to be within range of (0, 1]. |

<a id="preprocessing-quantize-scalar-quantizer"></a>
### preprocessing::quantize::scalar::quantizer

Defines and stores scalar for quantisation upon training

The quantization is performed by a linear mapping of an interval in the float data type to the full range of the quantized int type.

```cpp
template <typename T>
struct quantizer {
  T min_;
  T max_;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `min_` | `T` | Minimum value of the quantization range. |
| `max_` | `T` | Maximum value of the quantization range. |

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

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::host_matrix_view<const double, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<double>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
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

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<double, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<double>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
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

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::host_matrix_view<const float, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
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

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<float, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
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

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::host_matrix_view<const half, int64_t> dataset,
raft::host_matrix_view<int8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<half>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
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

Perform inverse quantization step on previously quantized dataset

```cpp
void inverse_transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::host_matrix_view<const int8_t, int64_t> dataset,
raft::host_matrix_view<half, int64_t> out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<half>&`](/api-reference/cpp-api-preprocessing-quantize-scalar#preprocessing-quantize-scalar-quantizer) | a scalar quantizer |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<half, int64_t>` | a row-major matrix view on host |

**Returns**

`void`
