---
slug: api-reference/cpp-api-core-dataset-sq
---

# SQ

_Source header: `cuvs/core/dataset/sq.hpp`_

## Types

<a id="sq-spec"></a>
### sq_spec

Scalar-quantized dataset specification: every component of a vector becomes one code, so the

manifest holds the quantizer and the dimension is still a property of the data. The manifest also holds the bound on the reconstruction error of the codes, which is a property of the dataset rather than of the quantizer.

```cpp
template <typename MathT, typename CodePolicy>
struct sq_spec;
```

<a id="sq-spec-manifest-view"></a>
### sq_spec::manifest_view

Borrowed quantizer parameters and the error bound of the dataset's codes.

The quantizer view base keeps the parameters directly accessible.

```cpp
template <typename QuantizerViewT>
struct manifest_view;
```

<a id="sq-spec-manifest-type"></a>
### sq_spec::manifest_type

Owns the quantizer, with either global or per-dimension bounds, and the error bound of the

dataset's codes.

```cpp
struct manifest_type;
```

## Scalar-quantized dataset construction

<a id="preprocessing-quantize-scalar-make-sq-dataset"></a>
### preprocessing::quantize::scalar::make_sq_dataset

Train and encode a device corpus with the selected quantization mode. For separate training and encoding, use train and transform.

```cpp
template <typename T>
inline device_sq_dataset<T, int64_t, T> make_sq_dataset(
raft::resources const& res,
params params,
raft::device_matrix_view<const T, int64_t> dataset,
bool compute_max_error = false);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources used for training and allocation. |
| `params` | in | `params` | Scalar quantizer parameters. |
| `dataset` | in | `raft::device_matrix_view<const T, int64_t>` | Device corpus to train on and encode. |
| `compute_max_error` | in | `bool` | Measure the dataset's maximum reconstruction error over every row.<br />Default: false, which leaves the manifest's max_error infinite.<br />Default: `false`. |

**Returns**

`inline device_sq_dataset<T, int64_t, T>`

Device dataset owning the quantized codes, the quantizer and the error bound.

**Additional overload:** `preprocessing::quantize::scalar::make_sq_dataset`

Train and encode a host corpus in GPU batches, returning device-resident int8 codes. Uses one quantizer across all batches. Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool on res to enable double buffering. All int8 codes must fit in device memory.

```cpp
device_sq_dataset<float, int64_t, float> make_sq_dataset(
raft::resources const& res,
params params,
raft::host_matrix_view<const float, int64_t> dataset,
bool compute_max_error = false);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources used for training, allocation, and transfer streams. |
| `params` | in | `params` | Scalar quantizer and host batching parameters. |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t>` | Row-major host corpus; pageable and pinned input are supported. |
| `compute_max_error` | in | `bool` | Measure the dataset's maximum reconstruction error over every row during the encoding pass. Default: false, which leaves the manifest's max_error infinite.<br />Default: `false`. |

**Returns**

`device_sq_dataset<float, int64_t, float>`

Device dataset owning the quantized codes, the quantizer and the error bound.

**Additional overload:** `preprocessing::quantize::scalar::make_sq_dataset`

Train and encode a host corpus in GPU batches, returning device-resident int8 codes. Uses one quantizer across all batches. Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool on res to enable double buffering. All int8 codes must fit in device memory.

```cpp
device_sq_dataset<double, int64_t, double> make_sq_dataset(
raft::resources const& res,
params params,
raft::host_matrix_view<const double, int64_t> dataset,
bool compute_max_error = false);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources used for training, allocation, and transfer streams. |
| `params` | in | `params` | Scalar quantizer and host batching parameters. |
| `dataset` | in | `raft::host_matrix_view<const double, int64_t>` | Row-major host corpus; pageable and pinned input are supported. |
| `compute_max_error` | in | `bool` | Measure the dataset's maximum reconstruction error over every row during the encoding pass. Default: false, which leaves the manifest's max_error infinite.<br />Default: `false`. |

**Returns**

`device_sq_dataset<double, int64_t, double>`

Device dataset owning the quantized codes, the quantizer and the error bound.

**Additional overload:** `preprocessing::quantize::scalar::make_sq_dataset`

Train and encode a host corpus in GPU batches, returning device-resident int8 codes. Uses one quantizer across all batches. Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool on res to enable double buffering. All int8 codes must fit in device memory.

```cpp
device_sq_dataset<half, int64_t, half> make_sq_dataset(
raft::resources const& res,
params params,
raft::host_matrix_view<const half, int64_t> dataset,
bool compute_max_error = false);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | Resources used for training, allocation, and transfer streams. |
| `params` | in | `params` | Scalar quantizer and host batching parameters. |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t>` | Row-major host corpus; pageable and pinned input are supported. |
| `compute_max_error` | in | `bool` | Measure the dataset's maximum reconstruction error over every row during the encoding pass. Default: false, which leaves the manifest's max_error infinite.<br />Default: `false`. |

**Returns**

`device_sq_dataset<half, int64_t, half>`

Device dataset owning the quantized codes, the quantizer and the error bound.
