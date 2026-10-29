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

<a id="preprocessing-quantize-binary-quantizer-view"></a>
### preprocessing::quantize::binary::quantizer_view

Non-owning view of a binary quantizer, with the constness semantics of an mdspan: `quantizer_view<T>` can modify the thresholds, `quantizer_view<const T>` cannot.

The threshold is either the vector of per-dimension thresholds (whose extent is the dimension), or just the dimension of the vectors, which stands for the zero quantizer: no thresholds are stored and zero is the threshold in every dimension. These are different quantizers even when the vector is empty.

```cpp
template <typename T,
typename ContainerPolicy =
raft::device_accessor<raft::device_container_policy<std::remove_const_t<T>>>>
struct quantizer_view {
  using vector_type;
  std::variant<uint32_t, vector_view_type> value;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `vector_type` | `using` | The owning vector of the thresholds, which defines the memory space of the view. |
| `value` | `std::variant<uint32_t, vector_view_type>` | The thresholds [dim], or the dimension for the zero quantizer. |

<a id="vector-type"></a>
### vector_type

The owning vector of the thresholds, which defines the memory space of the view.

```cpp
using vector_type      = raft::mdarray<std::remove_const_t<T>,
raft::vector_extent<int64_t>,
raft::layout_c_contiguous,
ContainerPolicy>;
```

<a id="preprocessing-quantize-binary-quantizer-view-quantizer-view"></a>
### preprocessing::quantize::binary::quantizer_view::quantizer_view

The zero quantizer of dimension zero.

```cpp
quantizer_view() = default;
```

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::quantizer_view::quantizer_view`

The zero quantizer of the given dimension (number of bits).

```cpp
explicit quantizer_view(uint32_t dim) noexcept : value;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` |  | `uint32_t` |  |

**Returns**

`explicit`

**Additional overload:** `preprocessing::quantize::binary::quantizer_view::quantizer_view`

The quantizer with the given thresholds; its dimension is the extent of the vector.

```cpp
explicit quantizer_view(vector_view_type thresholds) noexcept : value;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `thresholds` |  | `vector_view_type` |  |

**Returns**

`explicit`

<a id="preprocessing-quantize-binary-quantizer-view-dim"></a>
### preprocessing::quantize::binary::quantizer_view::dim

The number of dimensions (bits) of the quantized vectors.

```cpp
[[nodiscard]] auto dim() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="preprocessing-quantize-binary-quantizer-view-requires"></a>
### preprocessing::quantize::binary::quantizer_view::requires

A mutable view decays to the read-only one, the way an mdspan does.

```cpp
template <typename U>
requires(!std::is_const_v<T> && std::is_same_v<U, const T>)
operator quantizer_view<U, ContainerPolicy>()
const noexcept  // NOLINT(google-explicit-constructor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `!std::is_const_v<T> && std::is_same_v<U, const T>` |  |

**Returns**

`void`

<a id="preprocessing-quantize-binary-quantizer"></a>
### preprocessing::quantize::binary::quantizer

Store the threshold vector for quantization. In the binary::transform function, a bit is set if the corresponding element in the dataset vector is greater than the corresponding element in the threshold vector.

The threshold is either the vector of per-dimension thresholds (whose extent is the dimension), or just the dimension of the vectors, which stands for the zero quantizer: no thresholds are stored and zero is the threshold in every dimension. These are different quantizers even when the vector is empty. A default-constructed quantizer is the zero quantizer of dimension zero.

```cpp
template <typename T,
typename ContainerPolicy = raft::device_accessor<raft::device_container_policy<T>>>
struct quantizer {
  std::variant<uint32_t, threshold_type> value;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `value` | `std::variant<uint32_t, threshold_type>` | The thresholds [dim], or the dimension for the zero quantizer. |

<a id="preprocessing-quantize-binary-quantizer-quantizer"></a>
### preprocessing::quantize::binary::quantizer::quantizer

The zero quantizer of dimension zero.

```cpp
quantizer() = default;
```

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::binary::quantizer::quantizer`

The zero quantizer of the given dimension (number of bits).

```cpp
explicit quantizer(uint32_t dim) noexcept : value;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` |  | `uint32_t` |  |

**Returns**

`explicit`

**Additional overload:** `preprocessing::quantize::binary::quantizer::quantizer`

The quantizer owning the given thresholds; its dimension is the extent of the vector.

```cpp
explicit quantizer(threshold_type&& thresholds);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `thresholds` |  | `threshold_type&&` |  |

**Returns**

`explicit`

<a id="preprocessing-quantize-binary-quantizer-dim"></a>
### preprocessing::quantize::binary::quantizer::dim

The number of dimensions (bits) of the quantized vectors.

```cpp
[[nodiscard]] auto dim() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="preprocessing-quantize-binary-quantizer-view"></a>
### preprocessing::quantize::binary::quantizer::view

A non-owning view of the quantizer.

```cpp
[[nodiscard]] auto view() noexcept -> view_type;
```

**Returns**

`view_type`

**Additional overload:** `preprocessing::quantize::binary::quantizer::view`

A read-only non-owning view of the quantizer.

```cpp
[[nodiscard]] auto view() const noexcept -> const_view_type;
```

**Returns**

`const_view_type`

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
