---
slug: api-reference/c-api-preprocessing-quantize-scalar
---

# Scalar

_Source header: `cuvs/preprocessing/quantize/scalar.h`_

## C API for Scalar Quantizer

<a id="cuvsscalarquantizationmode"></a>
### cuvsScalarQuantizationMode

Select global or per-dimension scalar quantization bounds.

```c
enum cuvsScalarQuantizationMode {
  CUVS_SCALAR_GLOBAL = 0,
  CUVS_SCALAR_PER_DIMENSION = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_SCALAR_GLOBAL` | `0` |
| `CUVS_SCALAR_PER_DIMENSION` | `1` |

<a id="cuvsscalarquantizerparams"></a>
### cuvsScalarQuantizerParams

Scalar quantizer parameters. Allocate with cuvsScalarQuantizerParamsCreate.

```c
struct cuvsScalarQuantizerParams {
  float quantile;
  enum cuvsScalarQuantizationMode mode;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `quantile` | `float` | specifies how many outliers at top & bottom will be ignored needs to be within range of (0, 1] |
| `mode` | [`enum cuvsScalarQuantizationMode`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizationmode) | Quantization mode.<br />Defaults to CUVS_SCALAR_GLOBAL. |

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

The quantization is performed by a linear mapping of an interval in the float data type to the full range of the quantized int type. Allocate with cuvsScalarQuantizerCreate and release with cuvsScalarQuantizerDestroy; do not copy the handle's contents. The implementation also owns any per-dimension bounds. In global mode, min_ and max_ are the trained bounds. In per-dimension mode, they are numeric_limits&lt;T&gt;::min() and numeric_limits&lt;T&gt;::max() for the training data type T, and are not used for transformation.

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

De-allocate Scalar Quantizer after outstanding stream work completes.

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

Supports float16, float32, and float64 in global or per-dimension mode. Per-dimension mode applies the quantile independently to each dimension using every row. Successful retraining replaces the previous bounds; failed training preserves them. Complete outstanding stream work using the quantizer before retraining.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `params` | in | [`cuvsScalarQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizerparams) | configure scalar quantizer, e.g. quantile |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix |
| `quantizer` | in,out | [`cuvsScalarQuantizer_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizer) | scalar quantizer allocated by cuvsScalarQuantizerCreate |

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

Per-dimension mode requires device matrices with the same input type and width as training. Its output codes are int8 in [-127, 127]. The quantizer must outlive stream work using its bounds.

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

Note that depending on the chosen data types train dataset the conversion is not lossless. Per-dimension mode requires device matrices with the same output type and width as training. The quantizer must outlive stream work using its bounds.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `quantizer` | in | [`cuvsScalarQuantizer_t`](/api-reference/c-api-preprocessing-quantize-scalar#cuvsscalarquantizer) | a scalar quantizer |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix |
| `out` | out | `DLManagedTensor*` | a row-major host or device matrix |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
