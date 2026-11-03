---
slug: api-reference/c-api-preprocessing-quantize-rabitq
---

# Rabitq

_Source header: `cuvs/preprocessing/quantize/rabitq.h`_

## C API for RaBitQ Quantizer

<a id="cuvsrabitqrotatorkind"></a>
### cuvsRabitqRotatorKind

Random-rotation implementation used before quantization.

Both kinds produce an orthogonal transform of the padded working dimension; they differ only in cost and in the amount of state they carry.

```c
enum cuvsRabitqRotatorKind {
  CUVS_RABITQ_ROTATOR_KIND_MATMUL = 0,
  CUVS_RABITQ_ROTATOR_KIND_FHT_KAC = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_RABITQ_ROTATOR_KIND_MATMUL` | `0` |
| `CUVS_RABITQ_ROTATOR_KIND_FHT_KAC` | `1` |

<a id="cuvsrabitqdeltakind"></a>
### cuvsRabitqDeltaKind

Rule used to derive the per-vector reconstruction scale `delta`.

Only relevant for cuvsRabitqTransform / cuvsRabitqTransformResiduals, which emit `(codes, delta, vl)`; the full-factor entry points emit `(f_add, f_rescale, f_error)` instead and ignore this setting.

```c
enum cuvsRabitqDeltaKind {
  CUVS_RABITQ_DELTA_KIND_RECONSTRUCTION = 0,
  CUVS_RABITQ_DELTA_KIND_UNBIASED = 1,
  CUVS_RABITQ_DELTA_KIND_PLAIN = 2
};
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_RABITQ_DELTA_KIND_RECONSTRUCTION` | `0` |
| `CUVS_RABITQ_DELTA_KIND_UNBIASED` | `1` |
| `CUVS_RABITQ_DELTA_KIND_PLAIN` | `2` |

<a id="cuvsrabitqquantizerparams"></a>
### cuvsRabitqQuantizerParams

RaBitQ quantizer parameters.

These parameters fully describe both the rotator and the encoder: RaBitQ is data-oblivious, so none of them are learned from a dataset. This is why the factories below take a `dim` instead of a training dataset.

```c
struct cuvsRabitqQuantizerParams {
  uint32_t ex_bits;
  enum cuvsRabitqRotatorKind rotator;
  uint64_t seed;
  enum cuvsRabitqDeltaKind delta_mode;
  bool use_fast;
  uint32_t coarse_samples;
  uint32_t fine_samples;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `ex_bits` | `uint32_t` | Number of *extended* bits per dimension. The total code width is `ex_bits + 1` bits (one sign bit plus `ex_bits` magnitude bits), so `ex_bits = 0` is plain 1-bit RaBitQ and `ex_bits = 3` is the common 4-bit configuration.<br /><br />Possible value range: [0, 8]. |
| `rotator` | [`enum cuvsRabitqRotatorKind`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotatorkind) | Random-rotation implementation. |
| `seed` | `uint64_t` | Seed for the RNGs that draw the rotation state and the random probe vectors used to estimate the constant scaling factor. |
| `delta_mode` | [`enum cuvsRabitqDeltaKind`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqdeltakind) | Rule used to derive the per-vector `delta` / `vl` pair. |
| `use_fast` | `bool` | When true, every vector is quantized with a single pre-estimated scaling factor (see cuvsRabitqEncoderGetConstScalingFactor) instead of running the per-vector rescale search. This is substantially faster and is the recommended setting; disable it to trade build throughput for slightly tighter codes. |
| `coarse_samples` | `uint32_t` | Number of samples in the coarse stage of the per-vector rescale search. |
| `fine_samples` | `uint32_t` | Number of samples in the refinement stage of the per-vector rescale search. |

<a id="cuvsrabitqquantizerparamscreate"></a>
### cuvsRabitqQuantizerParamsCreate

Allocate RaBitQ quantizer params, and populate with default values

```c
cuvsError_t cuvsRabitqQuantizerParamsCreate(cuvsRabitqQuantizerParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsRabitqQuantizerParams_t*`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqquantizerparams) | cuvsRabitqQuantizerParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqquantizerparamsdestroy"></a>
### cuvsRabitqQuantizerParamsDestroy

De-allocate RaBitQ quantizer params

```c
cuvsError_t cuvsRabitqQuantizerParamsDestroy(cuvsRabitqQuantizerParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsRabitqQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqquantizerparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqrotator"></a>
### cuvsRabitqRotator

Opaque handle to the random rotation applied before quantization.

The rotator carries no learned information - its state is drawn from cuvsRabitqQuantizerParams::seed - but it *is* part of the code representation: codes produced with one rotator are only comparable to queries transformed by the very same rotator. There is deliberately no serialization API, so the caller owns this handle and is responsible for keeping it alongside the codes, by copying the buffers returned by cuvsRabitqRotatorGetState into its own index format and restoring them with cuvsRabitqRotatorMakeFromState. Re-running cuvsRabitqRotatorMake with the same params reproduces them only on the same GPU model and library version.

A rotator is an independent peer of the encoder: neither contains the other, and a single rotator may be shared by several encoders.

```c
typedef struct {
  uintptr_t addr;
} cuvsRabitqRotator;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |

<a id="cuvsrabitqrotatorcreate"></a>
### cuvsRabitqRotatorCreate

Allocate a RaBitQ rotator handle

```c
cuvsError_t cuvsRabitqRotatorCreate(cuvsRabitqRotator_t* rotator);
```

The handle starts out empty; populate it with cuvsRabitqRotatorMake, cuvsRabitqRotatorMakeFromState or cuvsRabitqQuantizerMake before use.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `rotator` | in | [`cuvsRabitqRotator_t*`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | cuvsRabitqRotator_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqrotatordestroy"></a>
### cuvsRabitqRotatorDestroy

De-allocate a RaBitQ rotator, freeing its device buffers

```c
cuvsError_t cuvsRabitqRotatorDestroy(cuvsRabitqRotator_t rotator);
```

Invalidates every tensor previously filled in by cuvsRabitqRotatorGetState.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `rotator` | in | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqencoder"></a>
### cuvsRabitqEncoder

Opaque handle to the (data-independent) configuration used to quantize rotated residuals.

RaBitQ needs no training data, so this is plain metadata plus the single scalar estimated at init time. Like the rotator it is owned by the caller and has no serialization API - it is cheap to rebuild from the params and `dim`.

```c
typedef struct {
  uintptr_t addr;
} cuvsRabitqEncoder;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |

<a id="cuvsrabitqencodercreate"></a>
### cuvsRabitqEncoderCreate

Allocate a RaBitQ encoder handle

```c
cuvsError_t cuvsRabitqEncoderCreate(cuvsRabitqEncoder_t* encoder);
```

The handle starts out empty; populate it with cuvsRabitqEncoderMake or cuvsRabitqQuantizerMake before use.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `encoder` | in | [`cuvsRabitqEncoder_t*`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | cuvsRabitqEncoder_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqencoderdestroy"></a>
### cuvsRabitqEncoderDestroy

De-allocate a RaBitQ encoder

```c
cuvsError_t cuvsRabitqEncoderDestroy(cuvsRabitqEncoder_t encoder);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `encoder` | in | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqrotatormake"></a>
### cuvsRabitqRotatorMake

Initializes a RaBitQ rotator.

```c
cuvsError_t cuvsRabitqRotatorMake(cuvsResources_t res,
cuvsRabitqQuantizerParams_t params,
int64_t dim,
cuvsRabitqRotator_t rotator);
```

No training data is required: the rotation is drawn from cuvsRabitqQuantizerParams::seed alone, which is why this takes a `dim` rather than a dataset. The rotator owns its device buffers; keep it alive for as long as the codes it produced are in use.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `params` | in | [`cuvsRabitqQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqquantizerparams) | configure the rotator, e.g. kind and seed |
| `dim` | in | `int64_t` | dimensionality of the un-rotated input vectors |
| `rotator` | out | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | an allocated rotator handle to initialize |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqrotatormakefromstate"></a>
### cuvsRabitqRotatorMakeFromState

Initializes a RaBitQ rotator from saved state.

```c
cuvsError_t cuvsRabitqRotatorMakeFromState(cuvsResources_t res,
enum cuvsRabitqRotatorKind kind,
DLManagedTensor* state,
cuvsRabitqRotator_t rotator);
```

The inverse of cuvsRabitqRotatorGetKind + cuvsRabitqRotatorGetState: restores a rotator persisted next to the codes it produced. `state` must have the shape and dtype cuvsRabitqRotatorGetState documents for `kind`, and the padded dimension is taken from it. It is copied, so it may live in device or host memory and may be freed once this returns.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `kind` | in | [`enum cuvsRabitqRotatorKind`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotatorkind) | the rotation implementation `state` belongs to |
| `state` | in | `DLManagedTensor*` | the saved rotator state |
| `rotator` | out | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | an allocated rotator handle to initialize |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqencodermake"></a>
### cuvsRabitqEncoderMake

Initializes a RaBitQ encoder.

```c
cuvsError_t cuvsRabitqEncoderMake(cuvsResources_t res,
cuvsRabitqQuantizerParams_t params,
int64_t dim,
cuvsRabitqEncoder_t encoder);
```

No training data is required, which is why this takes a `dim` rather than a dataset. When cuvsRabitqQuantizerParams::use_fast is set, this estimates the constant scaling factor on the GPU, which is the only non-trivial work performed here.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `params` | in | [`cuvsRabitqQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqquantizerparams) | configure the encoder, e.g. ex_bits |
| `dim` | in | `int64_t` | dimensionality of the un-rotated input vectors |
| `encoder` | out | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | an allocated encoder handle to initialize |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqquantizermake"></a>
### cuvsRabitqQuantizerMake

Initializes a RaBitQ quantizer: a rotator and an encoder from the same parameters.

```c
cuvsError_t cuvsRabitqQuantizerMake(cuvsResources_t res,
cuvsRabitqQuantizerParams_t params,
int64_t dim,
cuvsRabitqRotator_t rotator,
cuvsRabitqEncoder_t encoder);
```

Equivalent to calling cuvsRabitqRotatorMake and cuvsRabitqEncoderMake with the same arguments. No training data is required.

There is deliberately no opaque quantizer handle: the C++ `quantizer` struct is only a composition of the two independent peers - it adds no state and no behaviour of its own - so two out-parameters are the honest C spelling of it.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `params` | in | [`cuvsRabitqQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqquantizerparams) | configure both components |
| `dim` | in | `int64_t` | dimensionality of the un-rotated input vectors |
| `rotator` | out | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | an allocated rotator handle to initialize |
| `encoder` | out | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | an allocated encoder handle to initialize |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqrotate"></a>
### cuvsRabitqRotate

Applies the random rotation to already-padded rows.

```c
cuvsError_t cuvsRabitqRotate(cuvsResources_t res,
cuvsRabitqRotator_t rotator,
DLManagedTensor* in,
DLManagedTensor* out);
```

Both `in` and `out` must be `n_rows x padded_dim` row-major float32 device matrices (see cuvsRabitqRotatorGetPaddedDim); this entry point does not pad. In-place operation (`in` and `out` pointing at the same buffer) is supported by ::CUVS_RABITQ_ROTATOR_KIND_FHT_KAC only.

Use this to rotate queries or externally computed centroids so that they live in the same rotated space as the codes.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `rotator` | in | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | a RaBitQ rotator |
| `in` | in | `DLManagedTensor*` | a row-major `n_rows x padded_dim` float32 device matrix |
| `out` | out | `DLManagedTensor*` | a row-major `n_rows x padded_dim` float32 device matrix |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqtransform"></a>
### cuvsRabitqTransform

Quantizes a raw dataset, emitting per-vector `(delta, vl)` factors.

```c
cuvsError_t cuvsRabitqTransform(cuvsResources_t res,
cuvsRabitqEncoder_t encoder,
cuvsRabitqRotator_t rotator,
DLManagedTensor* dataset,
DLManagedTensor* centroid,
DLManagedTensor* codes,
DLManagedTensor* delta,
DLManagedTensor* vl);
```

Performs the whole chain: zero-pad `dim -> padded_dim`, rotate, optionally subtract the rotated centroid, then quantize the residuals. One code element is written per padded dimension; each holds an unsigned `cuvsRabitqQuantizerParams::ex_bits + 1` bit value, so `codes` must be `n_rows x padded_dim`.

Rows are processed in internal batches, so the temporary float buffer holding the rotated residuals, allocated from the workspace resource, covers one batch rather than the whole dataset (a pool workspace resource is recommended).

`dataset` may also be in host memory (kDLCPU). Its rows are then copied to the device one batch at a time, and are no longer read once the call returns. Pinned (kDLCUDAHost) and managed tensors are device-accessible and are read in place. `centroid` and the outputs must be on device.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `encoder` | in | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | a RaBitQ encoder |
| `rotator` | in | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | the rotator whose padded dim matches the encoder |
| `dataset` | in | `DLManagedTensor*` | a row-major `n_rows x dim` float32 host or device matrix |
| `centroid` | in | `DLManagedTensor*` | optional un-rotated `dim` float32 device vector; when non-NULL, codes are computed on the residuals w.r.t. this centroid. Pass NULL to skip centering. |
| `codes` | out | `DLManagedTensor*` | a row-major `n_rows x padded_dim` device matrix of uint8 or uint16 codes |
| `delta` | out | `DLManagedTensor*` | per-vector scale, an `n_rows` float32 device vector |
| `vl` | out | `DLManagedTensor*` | per-vector offset (`delta * -(2^ex_bits - 0.5)`), an `n_rows` float32 device vector |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqtransformfull"></a>
### cuvsRabitqTransformFull

Quantizes a raw dataset, emitting the full `(f_add, f_rescale, f_error)` factor triplet.

```c
cuvsError_t cuvsRabitqTransformFull(cuvsResources_t res,
cuvsRabitqEncoder_t encoder,
cuvsRabitqRotator_t rotator,
DLManagedTensor* dataset,
DLManagedTensor* centroid,
DLManagedTensor* codes,
DLManagedTensor* factors);
```

Same chain as cuvsRabitqTransform, but the per-vector factors are the ones used for approximate-distance estimation during search: `factors[i] = { f_add, f_rescale, f_error }`.

Rows are processed in internal batches, so the temporary float buffer holding the rotated residuals, allocated from the workspace resource, covers one batch rather than the whole dataset (a pool workspace resource is recommended). As for cuvsRabitqTransform, `dataset` may also be in host memory.

The rotated centroid is an internal temporary as well. A search implementation that needs it can reproduce it exactly by zero-padding `centroid` to `padded_dim` and calling cuvsRabitqRotate with the same rotator.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `encoder` | in | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | a RaBitQ encoder |
| `rotator` | in | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | the rotator whose padded dim matches the encoder |
| `dataset` | in | `DLManagedTensor*` | a row-major `n_rows x dim` float32 host or device matrix |
| `centroid` | in | `DLManagedTensor*` | optional un-rotated `dim` float32 device vector; treated as zero when NULL |
| `codes` | out | `DLManagedTensor*` | a row-major `n_rows x padded_dim` device matrix of uint8 or uint16 codes |
| `factors` | out | `DLManagedTensor*` | a row-major `n_rows x 3` float32 device matrix |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqtransformresiduals"></a>
### cuvsRabitqTransformResiduals

Quantizes pre-rotated residuals, emitting per-vector `(delta, vl)` factors.

```c
cuvsError_t cuvsRabitqTransformResiduals(cuvsResources_t res,
cuvsRabitqEncoder_t encoder,
DLManagedTensor* residuals,
DLManagedTensor* codes,
DLManagedTensor* delta,
DLManagedTensor* vl);
```

The rotator is not needed here: `residuals` must already be rotated, padded to the encoder's padded dim and centered (centroid subtracted). Use this when the residuals are produced elsewhere, e.g. by an IVF build that has already rotated its data, or by cuvsRabitqRotate plus a subtraction of your own.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `encoder` | in | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | a RaBitQ encoder |
| `residuals` | in | `DLManagedTensor*` | a row-major `n_rows x padded_dim` float32 device matrix, rotated and centered |
| `codes` | out | `DLManagedTensor*` | a row-major `n_rows x padded_dim` device matrix of uint8 or uint16 codes |
| `delta` | out | `DLManagedTensor*` | per-vector scale, an `n_rows` float32 device vector |
| `vl` | out | `DLManagedTensor*` | per-vector offset (`delta * -(2^ex_bits - 0.5)`), an `n_rows` float32 device vector |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqtransformresidualsfull"></a>
### cuvsRabitqTransformResidualsFull

Quantizes pre-rotated residuals, emitting the full factor triplet.

```c
cuvsError_t cuvsRabitqTransformResidualsFull(cuvsResources_t res,
cuvsRabitqEncoder_t encoder,
DLManagedTensor* residuals,
DLManagedTensor* rotated_centroid,
DLManagedTensor* codes,
DLManagedTensor* factors);
```

As above, `residuals` must already be rotated, padded to the encoder's padded dim and centered. The full-factor formula also reads the centroid, which must be supplied *in the rotated, padded space* (run cuvsRabitqRotate on it with the same rotator that produced the residuals); pass NULL to treat it as zero.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `encoder` | in | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | a RaBitQ encoder |
| `residuals` | in | `DLManagedTensor*` | a row-major `n_rows x padded_dim` float32 device matrix, rotated and centered |
| `rotated_centroid` | in | `DLManagedTensor*` | optional rotated, padded centroid (a `padded_dim` float32 device vector); treated as zero when NULL |
| `codes` | out | `DLManagedTensor*` | a row-major `n_rows x padded_dim` device matrix of uint8 or uint16 codes |
| `factors` | out | `DLManagedTensor*` | a row-major `n_rows x 3` float32 device matrix |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqrotatorgetkind"></a>
### cuvsRabitqRotatorGetKind

Get which random-rotation implementation a rotator uses.

```c
cuvsError_t cuvsRabitqRotatorGetKind(cuvsRabitqRotator_t rotator,
enum cuvsRabitqRotatorKind* kind);
```

Together with cuvsRabitqRotatorGetPaddedDim and cuvsRabitqRotatorGetState this is the rotator's full identity. Because RaBitQ has no serialization API, reading these three back out is how a C caller persists the rotator next to the codes it produced; cuvsRabitqRotatorMakeFromState restores it.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `rotator` | in | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | a RaBitQ rotator |
| `kind` | out | [`enum cuvsRabitqRotatorKind*`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotatorkind) | the rotation implementation |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqrotatorgetpaddeddim"></a>
### cuvsRabitqRotatorGetPaddedDim

Get the working dimension of a rotator: `dim` rounded up to a multiple of 32.

```c
cuvsError_t cuvsRabitqRotatorGetPaddedDim(cuvsRabitqRotator_t rotator,
int64_t* padded_dim);
```

Both transforms operate purely on the padded width, so the original unpadded `dim` is not part of the rotator.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `rotator` | in | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | a RaBitQ rotator |
| `padded_dim` | out | `int64_t*` | the working dimension |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqrotatorgetstate"></a>
### cuvsRabitqRotatorGetState

Get the device-side state of a rotator.

```c
cuvsError_t cuvsRabitqRotatorGetState(cuvsRabitqRotator_t rotator,
DLManagedTensor* state);
```

The shape and dtype depend on cuvsRabitqRotatorGetKind:

- ::CUVS_RABITQ_ROTATOR_KIND_MATMUL fills in the `padded_dim x padded_dim` row-major float32 rotation matrix `R`; cuvsRabitqRotate maps each row `x` to `R x`.
- ::CUVS_RABITQ_ROTATOR_KIND_FHT_KAC fills in the packed sign bits of the four Kac's-walk passes, a `4 * padded_dim / 8` element uint8 vector.

The returned DLManagedTensor is a **non-owning view**: the device buffer stays owned by `rotator` and becomes dangling once cuvsRabitqRotatorDestroy is called on it. Copy the contents out if you need them to outlive the rotator. Only the tensor's own shape/stride metadata is allocated here, and it is released by invoking the tensor's `deleter`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `rotator` | in | [`cuvsRabitqRotator_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqrotator) | a RaBitQ rotator |
| `state` | out | `DLManagedTensor*` | a view of the rotator's device state |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqencodergetdim"></a>
### cuvsRabitqEncoderGetDim

Get the dimensionality of the un-rotated input vectors an encoder expects.

```c
cuvsError_t cuvsRabitqEncoderGetDim(cuvsRabitqEncoder_t encoder, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `encoder` | in | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | a RaBitQ encoder |
| `dim` | out | `int64_t*` | dimensionality of the un-rotated input vectors |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqencodergetpaddeddim"></a>
### cuvsRabitqEncoderGetPaddedDim

Get the working dimension of an encoder: `dim` rounded up to a multiple of 32.

```c
cuvsError_t cuvsRabitqEncoderGetPaddedDim(cuvsRabitqEncoder_t encoder,
int64_t* padded_dim);
```

This is the number of code elements emitted per vector, i.e. the second extent that the `codes` tensor must have.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `encoder` | in | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | a RaBitQ encoder |
| `padded_dim` | out | `int64_t*` | the working dimension |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsrabitqencodergetconstscalingfactor"></a>
### cuvsRabitqEncoderGetConstScalingFactor

Get the scaling factor shared by all vectors on the fast path.

```c
cuvsError_t cuvsRabitqEncoderGetConstScalingFactor(cuvsRabitqEncoder_t encoder,
float* const_scaling_factor);
```

Estimated from random Gaussian probe vectors when the encoder was made. Left at 1.0 (and unused) when cuvsRabitqQuantizerParams::use_fast is false.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `encoder` | in | [`cuvsRabitqEncoder_t`](/api-reference/c-api-preprocessing-quantize-rabitq#cuvsrabitqencoder) | a RaBitQ encoder |
| `const_scaling_factor` | out | `float*` | the shared scaling factor |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
