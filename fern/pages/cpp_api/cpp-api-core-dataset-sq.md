---
slug: api-reference/cpp-api-core-dataset-sq
---

# SQ

_Source header: `cuvs/core/dataset/sq.hpp`_

## Types

<a id="sq-spec"></a>
### sq_spec

Scalar-quantized dataset specification: every component of a vector becomes one code, so the

manifest holds the quantizer and the dimension is still a property of the data.

```cpp
template <typename MathT, typename CodePolicy>
struct sq_spec;
```

## Scalar-quantized dataset construction

<a id="preprocessing-quantize-scalar-build"></a>
### preprocessing::quantize::scalar::build

Train and encode a device corpus with the selected range mode. Reuse result.const_manifest_view() with transform for query batches. For separate training/encoding, use train and transform.

```cpp
template <typename T>
inline device_sq_dataset<T, int64_t, T> build(raft::resources const& res,
params params,
raft::device_matrix_view<const T, int64_t> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources used for training and allocation. |
| `params` | in | `params` | Scalar quantizer parameters. |
| `dataset` | in | `raft::device_matrix_view<const T, int64_t>` | Device corpus to train on and encode. |

**Returns**

`inline device_sq_dataset<T, int64_t, T>`

Device dataset owning the quantized codes and quantizer.

**Additional overload:** `preprocessing::quantize::scalar::build`

Train and encode a host corpus in GPU batches, returning device-resident int8 codes. Uses one quantizer across all batches and measures the maximum error over every row. Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool on res to enable double buffering. All int8 codes must fit in device memory.

```cpp
device_sq_dataset<float, int64_t, float> build(
raft::resources const& res, params params, raft::host_matrix_view<const float, int64_t> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources used for training, allocation, and transfer streams. |
| `params` | in | `params` | Scalar quantizer and host batching parameters. |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | Row-major host corpus; pageable and pinned input are supported. |

**Returns**

`device_sq_dataset<float, int64_t, float>`

Device dataset owning the quantized codes and quantizer.

**Additional overload:** `preprocessing::quantize::scalar::build`

Train and encode a host corpus in GPU batches, returning device-resident int8 codes. Uses one quantizer across all batches and measures the maximum error over every row. Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool on res to enable double buffering. All int8 codes must fit in device memory.

```cpp
device_sq_dataset<double, int64_t, double> build(
raft::resources const& res, params params, raft::host_matrix_view<const double, int64_t> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources used for training, allocation, and transfer streams. |
| `params` | in | `params` | Scalar quantizer and host batching parameters. |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | Row-major host corpus; pageable and pinned input are supported. |

**Returns**

`device_sq_dataset<double, int64_t, double>`

Device dataset owning the quantized codes and quantizer.

**Additional overload:** `preprocessing::quantize::scalar::build`

Train and encode a host corpus in GPU batches, returning device-resident int8 codes. Uses one quantizer across all batches and measures the maximum error over every row. Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool on res to enable double buffering. All int8 codes must fit in device memory.

```cpp
device_sq_dataset<half, int64_t, half> build(raft::resources const& res,
params params,
raft::host_matrix_view<const half, int64_t> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources used for training, allocation, and transfer streams. |
| `params` | in | `params` | Scalar quantizer and host batching parameters. |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | Row-major host corpus; pageable and pinned input are supported. |

**Returns**

`device_sq_dataset<half, int64_t, half>`

Device dataset owning the quantized codes and quantizer.
