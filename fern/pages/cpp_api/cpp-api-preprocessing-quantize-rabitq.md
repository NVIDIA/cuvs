---
slug: api-reference/cpp-api-preprocessing-quantize-rabitq
---

# Rabitq

_Source header: `cuvs/preprocessing/quantize/rabitq.hpp`_

## RaBitQ quantizer utilities

<a id="preprocessing-quantize-rabitq-rotator-kind"></a>
### preprocessing::quantize::rabitq::rotator_kind

Random-rotation implementation used before quantization.

Both kinds produce an orthogonal transform of the padded working dimension; they differ only in cost and in the amount of state they carry.

```cpp
enum class rotator_kind : uint8_t {
  matmul,
  fht_kac
};
```

**Values**

| Name | Value |
| --- | --- |
| `matmul` | `` |
| `fht_kac` | `` |

<a id="preprocessing-quantize-rabitq-delta-kind"></a>
### preprocessing::quantize::rabitq::delta_kind

Rule used to derive the per-vector reconstruction scale `delta`.

Only relevant for the `(codes, delta, vl)` output mode; the full-factor mode emits `(f_add, f_rescale, f_error)` instead and ignores this setting.

In the formulas below, `code` means the signed levels `c - (2^ex_bits - 0.5)` of the stored code values `c`, not `c` itself.

```cpp
enum class delta_kind : uint8_t {
  reconstruction = 0,
  unbiased = 1,
  plain = 2
};
```

**Values**

| Name | Value | Description |
| --- | --- | --- |
| `reconstruction` | `0` | `delta = (\|residual\| / \|code\|) * cos(residual, code)`; minimizes reconstruction error. |
| `unbiased` | `1` | `delta = (\|residual\| / \|code\|) / cos(residual, code)`; unbiased inner-product estimate. |
| `plain` | `2` | `delta = \|residual\| / \|code\|`; preserves the residual norm, no angular correction. |

<a id="preprocessing-quantize-rabitq-params"></a>
### preprocessing::quantize::rabitq::params

RaBitQ quantizer parameters.

These parameters fully describe the quantizer: RaBitQ is data-oblivious, so none of them are learned from a dataset (see make_quantizer).

```cpp
struct params {
  uint32_t ex_bits;
  rotator_kind rotator;
  uint64_t seed;
  delta_kind delta_mode;
  bool use_fast;
  uint32_t coarse_samples;
  uint32_t fine_samples;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `ex_bits` | `uint32_t` | Number of *extended* bits per dimension. The total code width is `ex_bits + 1` bits (one sign bit plus `ex_bits` magnitude bits), so `ex_bits = 0` is plain 1-bit RaBitQ and `ex_bits = 3` is the common 4-bit configuration. `ex_bits = 8` needs `uint16_t` codes; `uint8_t` codes are rejected.<br /><br />Possible value range: [0, 8]. |
| `rotator` | [`rotator_kind`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator-kind) | Random-rotation implementation. |
| `seed` | `uint64_t` | Seed for the RNGs that draw the rotation state and the random probe vectors used to estimate the constant scaling factor. |
| `delta_mode` | [`delta_kind`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-delta-kind) | Rule used to derive the per-vector `delta` / `vl` pair. |
| `use_fast` | `bool` | When true, every vector is quantized with a single pre-estimated scaling factor (`encoder::const_scaling_factor`) instead of running the per-vector rescale search. This is substantially faster and is the recommended setting; disable it to trade build throughput for slightly tighter codes. |
| `coarse_samples` | `uint32_t` | Number of samples in the coarse stage of the per-vector rescale search. |
| `fine_samples` | `uint32_t` | Number of samples in the refinement stage of the per-vector rescale search. |

<a id="preprocessing-quantize-rabitq-rotator-view"></a>
### preprocessing::quantize::rabitq::rotator_view

Non-owning view of a rotator, with the constness semantics of an mdspan: `rotator_view<T>` can modify the rotation state, `rotator_view<const T>` cannot.

```cpp
template <typename T = float>
struct rotator_view {
  rotator_kind kind;
  int64_t padded_dim;
  raft::device_matrix_view<T, int64_t, raft::row_major> rotation_matrix;
  raft::device_vector_view<flip_type, int64_t> flip_bits;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `kind` | [`rotator_kind`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator-kind) | Which implementation the state below belongs to. |
| `padded_dim` | `int64_t` | Working dimension: `dim` rounded up to a multiple of 32. |
| `rotation_matrix` | `raft::device_matrix_view<T, int64_t, raft::row_major>` | See rotator::rotation_matrix. |
| `flip_bits` | `raft::device_vector_view<flip_type, int64_t>` | See rotator::flip_bits. |

<a id="preprocessing-quantize-rabitq-rotator-view-requires"></a>
### preprocessing::quantize::rabitq::rotator_view::requires

A mutable view decays to the read-only one, the way an mdspan does.

```cpp
template <typename U>
requires(!std::is_const_v<T> && std::is_same_v<U, const T>)
operator rotator_view<U>() const noexcept  // NOLINT(google-explicit-constructor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `!std::is_const_v<T> && std::is_same_v<U, const T>` |  |

**Returns**

`void`

<a id="preprocessing-quantize-rabitq-rotator"></a>
### preprocessing::quantize::rabitq::rotator

Random rotation applied to the data before quantization.

The rotator carries no learned information - its state is drawn from `params::seed` - but it *is* part of the code representation: codes produced with one rotator are only comparable to queries transformed by the very same rotator. There is deliberately no serialization API here, so the caller owns this object and is responsible for keeping it alongside the codes, by embedding the buffers below in its own index format. Re-running make_rotator with the same `params` reproduces them only on the same GPU model and library version.

A rotator is an independent peer of the encoder: neither contains the other, and a single rotator may be shared by several encoders.

```cpp
template <typename T = float>
struct rotator {
  rotator_kind kind;
  int64_t padded_dim;
  raft::device_matrix<T, int64_t, raft::row_major> rotation_matrix;
  raft::device_vector<uint8_t, int64_t> flip_bits;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `kind` | [`rotator_kind`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator-kind) | Which implementation the state below belongs to. |
| `padded_dim` | `int64_t` | Working dimension: `dim` rounded up to a multiple of 32.<br /><br />This is the rotator's full identity alongside `kind` and the state below — both transforms operate purely on the padded width (`fht_kac` derives its segment size as `2^floor_log2(padded_dim)`), so the original unpadded `dim` is not part of the rotator. |
| `rotation_matrix` | `raft::device_matrix<T, int64_t, raft::row_major>` | `padded_dim x padded_dim` row-major random orthogonal matrix `R`; rotate() maps each row `x` to `R x`. Empty unless `kind == rotator_kind::matmul`. |
| `flip_bits` | `raft::device_vector<uint8_t, int64_t>` | Packed sign bits of the four Kac's-walk passes (`4 * padded_dim / 8` bytes). Empty unless `kind == rotator_kind::fht_kac`. |

<a id="preprocessing-quantize-rabitq-rotator-rotator"></a>
### preprocessing::quantize::rabitq::rotator::rotator

Construct an empty (uninitialized) rotator.

```cpp
explicit rotator(raft::resources const& res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |

**Returns**

`explicit`

<a id="preprocessing-quantize-rabitq-rotator-view"></a>
### preprocessing::quantize::rabitq::rotator::view

A non-owning view of the rotator.

```cpp
[[nodiscard]] auto view() noexcept -> view_type;
```

**Returns**

`view_type`

**Additional overload:** `preprocessing::quantize::rabitq::rotator::view`

A read-only non-owning view of the rotator.

```cpp
[[nodiscard]] auto view() const noexcept -> const_view_type;
```

**Returns**

`const_view_type`

<a id="preprocessing-quantize-rabitq-encoder"></a>
### preprocessing::quantize::rabitq::encoder

Stores the (data-independent) configuration used to quantize rotated residuals.

RaBitQ needs no training data, so this is plain metadata plus the single scalar estimated at init time. Like the rotator, it is owned by the caller and has no serialization API - it is cheap to rebuild from `params` and `dim`.

```cpp
struct encoder {
  params params_quantizer;
  float const_scaling_factor;
  int64_t dim;
  int64_t padded_dim;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `params_quantizer` | [`params`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-params) | Parameters used to build this encoder. |
| `const_scaling_factor` | `float` | Scaling factor shared by all vectors on the `params::use_fast` path, estimated from random Gaussian probe vectors at init time. Left at 1.0 (and unused) when `params::use_fast` is false. |
| `dim` | `int64_t` | Dimensionality of the un-rotated input vectors. |
| `padded_dim` | `int64_t` | Working dimension: `dim` rounded up to a multiple of 32. |

<a id="preprocessing-quantize-rabitq-quantizer-view"></a>
### preprocessing::quantize::rabitq::quantizer_view

Non-owning view of a quantizer, with the constness semantics of an mdspan.

The encoder holds no buffers, so it is its own view.

```cpp
template <typename T = float>
struct quantizer_view {
  rotator_view<T> rotator;
  rabitq::encoder encoder;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `rotator` | [`rotator_view<T>`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator-view) | Rotation applied before encoding. |
| `encoder` | [`rabitq::encoder`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) | Encoder applied to the rotated residuals. |

<a id="preprocessing-quantize-rabitq-quantizer-view-requires"></a>
### preprocessing::quantize::rabitq::quantizer_view::requires

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

<a id="preprocessing-quantize-rabitq-quantizer"></a>
### preprocessing::quantize::rabitq::quantizer

A RaBitQ quantizer: the rotator and the encoder applied after it.

The entry point for the common end-to-end flow. It composes the two peers and adds no state and no behaviour of its own. Pass `quantizer::encoder` and `quantizer::rotator` to transform().

```cpp
template <typename T = float>
struct quantizer {
  rabitq::rotator<T> rotator;
  rabitq::encoder encoder;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `rotator` | [`rabitq::rotator<T>`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) | Rotation applied before encoding. |
| `encoder` | [`rabitq::encoder`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) | Encoder applied to the rotated residuals. |

<a id="preprocessing-quantize-rabitq-quantizer-quantizer"></a>
### preprocessing::quantize::rabitq::quantizer::quantizer

Construct an empty (uninitialized) quantizer.

```cpp
explicit quantizer(raft::resources const& res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |

**Returns**

`explicit`

<a id="preprocessing-quantize-rabitq-quantizer-view"></a>
### preprocessing::quantize::rabitq::quantizer::view

A non-owning view of the quantizer.

```cpp
[[nodiscard]] auto view() noexcept -> view_type;
```

**Returns**

`view_type`

**Additional overload:** `preprocessing::quantize::rabitq::quantizer::view`

A read-only non-owning view of the quantizer.

```cpp
[[nodiscard]] auto view() const noexcept -> const_view_type;
```

**Returns**

`const_view_type`

<a id="preprocessing-quantize-rabitq-make-rotator"></a>
### preprocessing::quantize::rabitq::make_rotator

Initializes a RaBitQ rotator.

```cpp
template <typename T = float>
auto make_rotator(raft::resources const& res, params const& params, int64_t dim) -> rotator<T>;
```

No training data is required: the rotation is drawn from `params::seed` alone. The returned object owns its device buffers; keep it for as long as the codes it produced are in use (see rotator).

Usage example:

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | element type of the rotation matrix |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`params const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-params) | configure the rotator, e.g. kind and seed |
| `dim` | in | `int64_t` | dimensionality of the un-rotated input vectors |

**Returns**

[`rotator<T>`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator)

<a id="preprocessing-quantize-rabitq-make-encoder"></a>
### preprocessing::quantize::rabitq::make_encoder

Initializes a RaBitQ encoder.

```cpp
encoder make_encoder(raft::resources const& res, params const& params, int64_t dim);
```

No training data is required. When `params::use_fast` is set, this estimates the constant scaling factor on the GPU, which is the only non-trivial work performed here.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`params const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-params) | configure the encoder, e.g. ex_bits |
| `dim` | in | `int64_t` | dimensionality of the un-rotated input vectors |

**Returns**

[`encoder`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder)

<a id="preprocessing-quantize-rabitq-make-quantizer"></a>
### preprocessing::quantize::rabitq::make_quantizer

Initializes a RaBitQ quantizer: a rotator and an encoder from the same parameters.

```cpp
template <typename T = float>
auto make_quantizer(raft::resources const& res, params const& params, int64_t dim) -> quantizer<T>;
```

Equivalent to calling make_rotator() and make_encoder() with the same arguments. No training data is required.

Usage example:

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | element type of the rotation matrix |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`params const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-params) | configure both components |
| `dim` | in | `int64_t` | dimensionality of the un-rotated input vectors |

**Returns**

[`quantizer<T>`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-quantizer)

<a id="preprocessing-quantize-rabitq-rotate"></a>
### preprocessing::quantize::rabitq::rotate

Applies the random rotation to already-padded rows.

```cpp
void rotate(raft::resources const& res,
rotator<float> const& rotator,
raft::device_matrix_view<const float, int64_t, raft::row_major> in,
raft::device_matrix_view<float, int64_t, raft::row_major> out);
```

Both `in` and `out` must be `n_rows x rotator::padded_dim`; this entry point does not pad. In-place operation (`in.data_handle() == out.data_handle()`) is supported by `rotator_kind::fht_kac` only.

Use this to rotate queries or externally computed centroids so that they live in the same rotated space as the codes.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `rotator` | in | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) | a RaBitQ rotator |
| `in` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a row-major `n_rows x padded_dim` matrix view on device |
| `out` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | a row-major `n_rows x padded_dim` matrix view on device |

**Returns**

`void`

<a id="preprocessing-quantize-rabitq-transform"></a>
### preprocessing::quantize::rabitq::transform

Quantizes a raw dataset, emitting per-vector `(delta, vl)` factors.

```cpp
void transform(raft::resources const& res,
encoder const& encoder,
rotator<float> const& rotator,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::device_vector_view<const float, int64_t>> centroid,
raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
raft::device_vector_view<float, int64_t> delta,
raft::device_vector_view<float, int64_t> vl);
```

Performs the whole chain: zero-pad `dim -> padded_dim`, rotate, optionally subtract the rotated centroid, then quantize the residuals. One code element is written per padded dimension; each holds an unsigned `params::ex_bits + 1` bit value, so `codes` must be `n_rows x padded_dim`.

Rows are processed in internal batches. A temporary float buffer is used internally to store the rotated residuals. This is allocated from the workspace resource. (A pool workspace resource is recommended.)

`dataset` may also be on host. Its rows are then copied to the device one batch at a time into workspace buffers, and are no longer read once the call returns; `centroid` and the outputs are on device either way. Host input is bound by the host-to-device copy, which is about twice as fast from pinned memory (e.g. raft::make_pinned_matrix, passed as a raft::host_matrix_view) as from pageable memory. A stream pool on `res` (raft::resource::set_cuda_stream_pool) overlaps copying each batch with quantizing the previous one.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `encoder` | in | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) | a RaBitQ encoder |
| `rotator` | in | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) | the rotator whose `padded_dim` matches the encoder |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a row-major `n_rows x dim` matrix view on device or host |
| `centroid` | in | `std::optional<raft::device_vector_view<const float, int64_t>>` | optional un-rotated `dim` centroid on device; when set, codes are computed on the residuals w.r.t. this centroid |
| `codes` | out | `raft::device_matrix_view<uint8_t, int64_t, raft::row_major>` | a row-major `n_rows x padded_dim` matrix view on device |
| `delta` | out | `raft::device_vector_view<float, int64_t>` | per-vector scale, `n_rows` elements on device |
| `vl` | out | `raft::device_vector_view<float, int64_t>` | per-vector offset (`delta * -(2^ex_bits - 0.5)`), `n_rows` elements on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform`

```cpp
void transform(raft::resources const& res,
encoder const& encoder,
rotator<float> const& rotator,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::device_vector_view<const float, int64_t>> centroid,
raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
raft::device_vector_view<float, int64_t> delta,
raft::device_vector_view<float, int64_t> vl);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `encoder` |  | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) |  |
| `rotator` |  | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) |  |
| `dataset` |  | `raft::device_matrix_view<const float, int64_t, raft::row_major>` |  |
| `centroid` |  | `std::optional<raft::device_vector_view<const float, int64_t>>` |  |
| `codes` |  | `raft::device_matrix_view<uint16_t, int64_t, raft::row_major>` |  |
| `delta` |  | `raft::device_vector_view<float, int64_t>` |  |
| `vl` |  | `raft::device_vector_view<float, int64_t>` |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform`

```cpp
void transform(raft::resources const& res,
encoder const& encoder,
rotator<float> const& rotator,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::device_vector_view<const float, int64_t>> centroid,
raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
raft::device_vector_view<float, int64_t> delta,
raft::device_vector_view<float, int64_t> vl);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `encoder` |  | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) |  |
| `rotator` |  | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t, raft::row_major>` |  |
| `centroid` |  | `std::optional<raft::device_vector_view<const float, int64_t>>` |  |
| `codes` |  | `raft::device_matrix_view<uint8_t, int64_t, raft::row_major>` |  |
| `delta` |  | `raft::device_vector_view<float, int64_t>` |  |
| `vl` |  | `raft::device_vector_view<float, int64_t>` |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform`

```cpp
void transform(raft::resources const& res,
encoder const& encoder,
rotator<float> const& rotator,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::device_vector_view<const float, int64_t>> centroid,
raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
raft::device_vector_view<float, int64_t> delta,
raft::device_vector_view<float, int64_t> vl);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `encoder` |  | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) |  |
| `rotator` |  | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t, raft::row_major>` |  |
| `centroid` |  | `std::optional<raft::device_vector_view<const float, int64_t>>` |  |
| `codes` |  | `raft::device_matrix_view<uint16_t, int64_t, raft::row_major>` |  |
| `delta` |  | `raft::device_vector_view<float, int64_t>` |  |
| `vl` |  | `raft::device_vector_view<float, int64_t>` |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform`

Quantizes a raw dataset, emitting the full `(f_add, f_rescale, f_error)` factor triplet.

```cpp
void transform(raft::resources const& res,
encoder const& encoder,
rotator<float> const& rotator,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::device_vector_view<const float, int64_t>> centroid,
raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
raft::device_matrix_view<float, int64_t, raft::row_major> factors);
```

Same chain as the `(delta, vl)` overload, but the per-vector factors are the ones used for approximate-distance estimation during search: `factors[i] = { f_add, f_rescale, f_error }`.

Rows are processed in internal batches. A temporary float buffer is used internally to store the rotated residuals. This is allocated from the workspace resource. (A pool workspace resource is recommended.) As for the `(delta, vl)` overload, `dataset` may also be on host.

The rotated centroid is an internal temporary as well. A search implementation that needs it can reproduce it exactly by zero-padding `centroid` to `padded_dim` and calling rotate() with the same rotator.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `encoder` | in | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) | a RaBitQ encoder |
| `rotator` | in | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) | the rotator whose `padded_dim` matches the encoder |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a row-major `n_rows x dim` matrix view on device or host |
| `centroid` | in | `std::optional<raft::device_vector_view<const float, int64_t>>` | optional un-rotated `dim` centroid on device; treated as zero when unset |
| `codes` | out | `raft::device_matrix_view<uint8_t, int64_t, raft::row_major>` | a row-major `n_rows x padded_dim` matrix view on device |
| `factors` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | a row-major `n_rows x 3` matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform`

```cpp
void transform(raft::resources const& res,
encoder const& encoder,
rotator<float> const& rotator,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::device_vector_view<const float, int64_t>> centroid,
raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
raft::device_matrix_view<float, int64_t, raft::row_major> factors);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `encoder` |  | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) |  |
| `rotator` |  | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) |  |
| `dataset` |  | `raft::device_matrix_view<const float, int64_t, raft::row_major>` |  |
| `centroid` |  | `std::optional<raft::device_vector_view<const float, int64_t>>` |  |
| `codes` |  | `raft::device_matrix_view<uint16_t, int64_t, raft::row_major>` |  |
| `factors` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform`

```cpp
void transform(raft::resources const& res,
encoder const& encoder,
rotator<float> const& rotator,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::device_vector_view<const float, int64_t>> centroid,
raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
raft::device_matrix_view<float, int64_t, raft::row_major> factors);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `encoder` |  | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) |  |
| `rotator` |  | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t, raft::row_major>` |  |
| `centroid` |  | `std::optional<raft::device_vector_view<const float, int64_t>>` |  |
| `codes` |  | `raft::device_matrix_view<uint8_t, int64_t, raft::row_major>` |  |
| `factors` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform`

```cpp
void transform(raft::resources const& res,
encoder const& encoder,
rotator<float> const& rotator,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::device_vector_view<const float, int64_t>> centroid,
raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
raft::device_matrix_view<float, int64_t, raft::row_major> factors);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `encoder` |  | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) |  |
| `rotator` |  | [`rotator<float> const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-rotator) |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t, raft::row_major>` |  |
| `centroid` |  | `std::optional<raft::device_vector_view<const float, int64_t>>` |  |
| `codes` |  | `raft::device_matrix_view<uint16_t, int64_t, raft::row_major>` |  |
| `factors` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

<a id="preprocessing-quantize-rabitq-transform-residuals"></a>
### preprocessing::quantize::rabitq::transform_residuals

Quantizes pre-rotated residuals, emitting per-vector `(delta, vl)` factors.

```cpp
void transform_residuals(raft::resources const& res,
encoder const& encoder,
raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
raft::device_vector_view<float, int64_t> delta,
raft::device_vector_view<float, int64_t> vl);
```

The rotator is not needed here: `residuals` must already be rotated, padded to `encoder::padded_dim` and centered (centroid subtracted). Use this when the residuals are produced elsewhere, e.g. by an IVF build that has already rotated its data, and rotate() plus a subtraction of your own.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `encoder` | in | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) | a RaBitQ encoder |
| `residuals` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a row-major `n_rows x padded_dim` matrix view on device, rotated and centered |
| `codes` | out | `raft::device_matrix_view<uint8_t, int64_t, raft::row_major>` | a row-major `n_rows x padded_dim` matrix view on device |
| `delta` | out | `raft::device_vector_view<float, int64_t>` | per-vector scale, `n_rows` elements on device |
| `vl` | out | `raft::device_vector_view<float, int64_t>` | per-vector offset (`delta * -(2^ex_bits - 0.5)`), `n_rows` elements on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform_residuals`

```cpp
void transform_residuals(raft::resources const& res,
encoder const& encoder,
raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
raft::device_vector_view<float, int64_t> delta,
raft::device_vector_view<float, int64_t> vl);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `encoder` |  | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) |  |
| `residuals` |  | `raft::device_matrix_view<const float, int64_t, raft::row_major>` |  |
| `codes` |  | `raft::device_matrix_view<uint16_t, int64_t, raft::row_major>` |  |
| `delta` |  | `raft::device_vector_view<float, int64_t>` |  |
| `vl` |  | `raft::device_vector_view<float, int64_t>` |  |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform_residuals`

Quantizes pre-rotated residuals, emitting the full factor triplet.

```cpp
void transform_residuals(
raft::resources const& res,
encoder const& encoder,
raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
std::optional<raft::device_vector_view<const float, int64_t>> rotated_centroid,
raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
raft::device_matrix_view<float, int64_t, raft::row_major> factors);
```

As above, `residuals` must already be rotated, padded to `encoder::padded_dim` and centered. The full-factor formula also reads the centroid, which must be supplied *in the rotated, padded space* (rotate() it with the same rotator that produced the residuals); pass `std::nullopt` to treat it as zero.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `encoder` | in | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) | a RaBitQ encoder |
| `residuals` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a row-major `n_rows x padded_dim` matrix view on device, rotated and centered |
| `rotated_centroid` | in | `std::optional<raft::device_vector_view<const float, int64_t>>` | optional rotated, padded centroid (`padded_dim` elements on device) |
| `codes` | out | `raft::device_matrix_view<uint8_t, int64_t, raft::row_major>` | a row-major `n_rows x padded_dim` matrix view on device |
| `factors` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | a row-major `n_rows x 3` matrix view on device |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::rabitq::transform_residuals`

```cpp
void transform_residuals(
raft::resources const& res,
encoder const& encoder,
raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
std::optional<raft::device_vector_view<const float, int64_t>> rotated_centroid,
raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
raft::device_matrix_view<float, int64_t, raft::row_major> factors);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `encoder` |  | [`encoder const&`](/api-reference/cpp-api-preprocessing-quantize-rabitq#preprocessing-quantize-rabitq-encoder) |  |
| `residuals` |  | `raft::device_matrix_view<const float, int64_t, raft::row_major>` |  |
| `rotated_centroid` |  | `std::optional<raft::device_vector_view<const float, int64_t>>` |  |
| `codes` |  | `raft::device_matrix_view<uint16_t, int64_t, raft::row_major>` |  |
| `factors` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`
