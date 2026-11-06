---
slug: api-reference/c-api-preprocessing-quantize-scalar
---

# Scalar

_Source header: `cuvs/preprocessing/quantize/scalar.h`_

## C API for Scalar Quantizer

<a id="cuvsscalarquantizerparams"></a>
### cuvsScalarQuantizerParams

Scalar quantizer parameters.

```c
struct cuvsScalarQuantizerParams;
```

<a id="cuvsscalarquantizerparamscreate"></a>
### cuvsScalarQuantizerParamsCreate

Allocate Scalar Quantizer params, and populate with default values

```c
cuvsError_t cuvsScalarQuantizerParamsCreate(cuvsScalarQuantizerParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsScalarQuantizerParams_t*`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerparams) | cuvsScalarQuantizerParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizerparamsdestroy"></a>
### cuvsScalarQuantizerParamsDestroy

De-allocate Scalar Quantizer params

```c
cuvsError_t cuvsScalarQuantizerParamsDestroy(cuvsScalarQuantizerParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsScalarQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizer"></a>
### cuvsScalarQuantizer

Defines and stores scalar for quantisation upon training

The quantization is performed by a linear mapping of an interval in the float data type to the full range of the quantized int type.

```c
typedef struct {
  double min_;
  double max_;
} cuvsScalarQuantizer;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `min_` | `double` |  |
| `max_` | `double` |  |

<a id="cuvsscalarquantizercreate"></a>
### cuvsScalarQuantizerCreate

Allocate Scalar Quantizer and populate with default values

```c
cuvsError_t cuvsScalarQuantizerCreate(cuvsScalarQuantizer_t* quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsScalarQuantizer_t*`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizer) | cuvsScalarQuantizer_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizerdestroy"></a>
### cuvsScalarQuantizerDestroy

De-allocate Scalar Quantizer

```c
cuvsError_t cuvsScalarQuantizerDestroy(cuvsScalarQuantizer_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsScalarQuantizer_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizer) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizertrain"></a>
### cuvsScalarQuantizerTrain

Trains a scalar quantizer to be used later for quantizing the dataset.

```c
cuvsError_t cuvsScalarQuantizerTrain(cuvsResources_t res,
cuvsScalarQuantizerParams_t params,
DLManagedTensor* dataset,
cuvsScalarQuantizer_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `params` | in | [`cuvsScalarQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerparams) | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix |
| `quantizer` | out | [`cuvsScalarQuantizer_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizer) | trained scalar quantizer |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizertransform"></a>
### cuvsScalarQuantizerTransform

Applies quantization transform to given dataset

```c
cuvsError_t cuvsScalarQuantizerTransform(cuvsResources_t res,
cuvsScalarQuantizer_t quantizer,
DLManagedTensor* dataset,
DLManagedTensor* out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `quantizer` | in | [`cuvsScalarQuantizer_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizer) | a scalar quantizer |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix to transform |
| `out` | out | `DLManagedTensor*` | a row-major host or device matrix to store transformed data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizerinversetransform"></a>
### cuvsScalarQuantizerInverseTransform

Perform inverse quantization step on previously quantized dataset

```c
cuvsError_t cuvsScalarQuantizerInverseTransform(cuvsResources_t res,
cuvsScalarQuantizer_t quantizer,
DLManagedTensor* dataset,
DLManagedTensor* out);
```

Note that depending on the chosen data types train dataset the conversion is not lossless.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `quantizer` | in | [`cuvsScalarQuantizer_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizer) | a scalar quantizer |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix |
| `out` | out | `DLManagedTensor*` | a row-major host or device matrix |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizerperdim"></a>
### cuvsScalarQuantizerPerDim

Owns a scalar quantizer with separate min/max bounds for each dimension. The fields are managed by the lifecycle and training functions.

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsScalarQuantizerPerDim;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsscalarquantizerperdimcreate"></a>
### cuvsScalarQuantizerPerDimCreate

Allocate an untrained per-dimension scalar quantizer.

```c
cuvsError_t cuvsScalarQuantizerPerDimCreate(cuvsScalarQuantizerPerDim_t* quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | out | [`cuvsScalarQuantizerPerDim_t*`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerperdim) | Handle to allocate. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizerperdimdestroy"></a>
### cuvsScalarQuantizerPerDimDestroy

Release the quantizer and its device bounds after outstanding stream work completes.

```c
cuvsError_t cuvsScalarQuantizerPerDimDestroy(cuvsScalarQuantizerPerDim_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsScalarQuantizerPerDim_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerperdim) | Handle to destroy. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizerperdimtrain"></a>
### cuvsScalarQuantizerPerDimTrain

Train per-dimension bounds from every row of a host or device matrix. Supports float16, float32, and float64. The quantile is applied independently per dimension. Successful retraining replaces the previous bounds; failed training preserves them. Complete outstanding stream work using the quantizer before retraining.

```c
cuvsError_t cuvsScalarQuantizerPerDimTrain(cuvsResources_t res,
cuvsScalarQuantizerParams_t params,
DLManagedTensor* dataset,
cuvsScalarQuantizerPerDim_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | Resources used for training and allocation. |
| `params` | in | [`cuvsScalarQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerparams) | Scalar quantizer parameters. |
| `dataset` | in | `DLManagedTensor*` | Row-major host or device matrix. |
| `quantizer` | in,out | [`cuvsScalarQuantizerPerDim_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerperdim) | Handle created by cuvsScalarQuantizerPerDimCreate. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizerperdimtransform"></a>
### cuvsScalarQuantizerPerDimTransform

Quantize a device matrix using previously trained per-dimension bounds. The input type and width must match training. Output codes are int8, in [-127, 127]. The quantizer must outlive stream work using its bounds.

```c
cuvsError_t cuvsScalarQuantizerPerDimTransform(cuvsResources_t res,
cuvsScalarQuantizerPerDim_t quantizer,
DLManagedTensor* dataset,
DLManagedTensor* out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | Resources used for execution. |
| `quantizer` | in | [`cuvsScalarQuantizerPerDim_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerperdim) | Trained per-dimension scalar quantizer. |
| `dataset` | in | `DLManagedTensor*` | Row-major device matrix to encode. |
| `out` | out | `DLManagedTensor*` | Row-major device int8 matrix with the same shape. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsscalarquantizerperdiminversetransform"></a>
### cuvsScalarQuantizerPerDimInverseTransform

Reconstruct a device matrix from per-dimension scalar codes. The output type and width must match training. The quantizer must outlive stream work using its bounds.

```c
cuvsError_t cuvsScalarQuantizerPerDimInverseTransform(
cuvsResources_t res,
cuvsScalarQuantizerPerDim_t quantizer,
DLManagedTensor* dataset,
DLManagedTensor* out);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | Resources used for execution. |
| `quantizer` | in | [`cuvsScalarQuantizerPerDim_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerperdim) | Trained per-dimension scalar quantizer. |
| `dataset` | in | `DLManagedTensor*` | Row-major device int8 matrix to decode. |
| `out` | out | `DLManagedTensor*` | Row-major device matrix with the same shape, in the training data type. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
