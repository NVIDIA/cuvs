---
slug: api-reference/cpp-api-preprocessing-quantize-binary
---

# Binary

_Source header: `cuvs/preprocessing/quantize/binary.hpp`_

## Binary quantizer utilities

<a id="preprocessing-quantize-binary-bit-threshold"></a>
### preprocessing::quantize::binary::bit_threshold

quantizer algorithms. The mean and sampling_median thresholds are calculated separately for each dimension.

```cpp
enum class bit_threshold {
  zero,
  mean,
  sampling_median
};
```

**Values**

| Name | Value |
| --- | --- |
| `zero` | `` |
| `mean` | `` |
| `sampling_median` | `` |

<a id="preprocessing-quantize-binary-params"></a>
### preprocessing::quantize::binary::params

quantizer parameters.

```cpp
struct params {
  bit_threshold threshold;
  float sampling_ratio;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `threshold` | [`bit_threshold`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-bit-threshold) | Threshold method for binarization. |
| `sampling_ratio` | `float` | Specifies the sampling ratio. |

<a id="preprocessing-quantize-binary-quantizer"></a>
### preprocessing::quantize::binary::quantizer

Store the threshold vector for quantization. In the binary::transform function, a bit is set if the corresponding element in the dataset vector is greater than the corresponding element in the threshold vector.

```cpp
template <typename T>
struct quantizer {
  raft::device_vector<T, int64_t> threshold;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `threshold` | `raft::device_vector<T, int64_t>` | Threshold vector used for binarization. |

<a id="preprocessing-quantize-binary-quantizer-quantizer"></a>
### preprocessing::quantize::binary::quantizer::quantizer

Construct a quantizer with an empty threshold vector.

```cpp
quantizer(raft::resources const& res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |

**Returns**

`void`

<a id="preprocessing-quantize-binary-train"></a>
### preprocessing::quantize::binary::train

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |

**Returns**

[`quantizer<double>`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer)

**Additional overload:** `preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |

**Returns**

[`quantizer<double>`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer)

<a id="preprocessing-quantize-binary-transform"></a>
### preprocessing::quantize::binary::transform

Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
void transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::device_matrix_view<const double, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<double>&`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) | a binary quantizer |
| `dataset` | in | `raft::device_matrix_view<const double, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
void transform(raft::resources const& res,
const quantizer<double>& quantizer,
raft::host_matrix_view<const double, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<double>&`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) | a binary quantizer |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<uint8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |

**Returns**

[`quantizer<float>`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer)

**Additional overload:** `preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |

**Returns**

[`quantizer<float>`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer)

**Additional overload:** `preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) | a binary quantizer |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quantizer,
raft::host_matrix_view<const float, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) | a binary quantizer |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<uint8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |

**Returns**

[`quantizer<half>`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer)

**Additional overload:** `preprocessing::quantize::binary::train`

Initializes a binary quantizer to be used later for quantizing the dataset.

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
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-params) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |

**Returns**

[`quantizer<half>`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer)

**Additional overload:** `preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset.

```cpp
void transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::device_matrix_view<const half, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<half>&`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) | a binary quantizer |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t>` | a row-major matrix view on device |
| `out` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a row-major matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::transform`

Applies binary quantization transform to given dataset.

```cpp
void transform(raft::resources const& res,
const quantizer<half>& quantizer,
raft::host_matrix_view<const half, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quantizer` | in | [`const quantizer<half>&`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) | a binary quantizer |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | a row-major matrix view on host |
| `out` | out | `raft::host_matrix_view<uint8_t, int64_t>` | a row-major matrix view on host |

**Returns**

`void`

<a id="preprocessing-quantize-binary-deprecated"></a>
### preprocessing::quantize::binary::[[deprecated

[deprecated] Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::device_matrix_view<const double, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`"please create and specify a quantizer"`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::host_matrix_view<const double, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`"please create and specify a quantizer"`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`"please create and specify a quantizer"`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::host_matrix_view<const float, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`"please create and specify a quantizer"`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::device_matrix_view<const half, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`"please create and specify a quantizer"`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::[[deprecated`

[deprecated] Applies binary quantization transform to given dataset. If a dataset element is positive, set the corresponding bit to 1.

```cpp
[[deprecated("please create and specify a quantizer")]] void transform(
raft::resources const& res,
raft::host_matrix_view<const half, int64_t> dataset,
raft::host_matrix_view<uint8_t, int64_t> out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`"please create and specify a quantizer"`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer) |  |

**Returns**

`void`
