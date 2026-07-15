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
