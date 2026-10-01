---
slug: api-reference/c-api-preprocessing-quantize-binary
---

# Binary

_Source header: `cuvs/preprocessing/quantize/binary.h`_

## C API for Binary Quantizer

<a id="cuvsbinaryquantizerthreshold"></a>
### cuvsBinaryQuantizerThreshold

In the cuvsBinaryQuantizerTransform function, a bit is set if the corresponding element in the dataset vector is greater than the corresponding element in the threshold vector. The mean and sampling_median thresholds are calculated separately for each dimension.

```c
enum cuvsBinaryQuantizerThreshold {
  ZERO = 0,
  MEAN = 1,
  SAMPLING_MEDIAN = 2
};
```

**Values**

| Name | Value |
| --- | --- |
| `ZERO` | `0` |
| `MEAN` | `1` |
| `SAMPLING_MEDIAN` | `2` |

<a id="cuvsbinaryquantizerparams"></a>
### cuvsBinaryQuantizerParams

Binary quantizer parameters.

```c
struct cuvsBinaryQuantizerParams {
  /* * specifies the threshold to set a bit in cuvsBinaryQuantizerTransform */ enum cuvsBinaryQuantizerThreshold threshold;
  /* * specifies the sampling ratio */ float sampling_ratio;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `threshold` | [`/* * specifies the threshold to set a bit in cuvsBinaryQuantizerTransform */ enum cuvsBinaryQuantizerThreshold`](/api-reference/c-api-preprocessing-quantize-binary#cuvsbinaryquantizerthreshold) |  |
| `sampling_ratio` | `/* * specifies the sampling ratio */ float` |  |

<a id="cuvsbinaryquantizerparamscreate"></a>
### cuvsBinaryQuantizerParamsCreate

Allocate Binary Quantizer params, and populate with default values

```c
cuvsError_t cuvsBinaryQuantizerParamsCreate(cuvsBinaryQuantizerParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsBinaryQuantizerParams_t*`](/api-reference/c-api-preprocessing-quantize-binary#cuvsbinaryquantizerparams) | cuvsBinaryQuantizerParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbinaryquantizerparamsdestroy"></a>
### cuvsBinaryQuantizerParamsDestroy

De-allocate Binary Quantizer params

```c
cuvsError_t cuvsBinaryQuantizerParamsDestroy(cuvsBinaryQuantizerParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsBinaryQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-binary#cuvsbinaryquantizerparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbinaryquantizer"></a>
### cuvsBinaryQuantizer

Defines and stores threshold for quantization upon training

The quantization is performed by a linear mapping of an interval in the float data type to the full range of the quantized int type.

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsBinaryQuantizer;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsbinaryquantizercreate"></a>
### cuvsBinaryQuantizerCreate

Allocate Binary Quantizer and populate with default values

```c
cuvsError_t cuvsBinaryQuantizerCreate(cuvsBinaryQuantizer_t* quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsBinaryQuantizer_t*`](/api-reference/c-api-preprocessing-quantize-binary#cuvsbinaryquantizer) | cuvsBinaryQuantizer_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbinaryquantizerdestroy"></a>
### cuvsBinaryQuantizerDestroy

De-allocate Binary Quantizer

```c
cuvsError_t cuvsBinaryQuantizerDestroy(cuvsBinaryQuantizer_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsBinaryQuantizer_t`](/api-reference/c-api-preprocessing-quantize-binary#cuvsbinaryquantizer) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbinaryquantizertrain"></a>
### cuvsBinaryQuantizerTrain

Trains a binary quantizer to be used later for quantizing the dataset.

```c
cuvsError_t cuvsBinaryQuantizerTrain(cuvsResources_t res,
cuvsBinaryQuantizerParams_t params,
DLManagedTensor* dataset,
cuvsBinaryQuantizer_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `params` | in | [`cuvsBinaryQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-binary#cuvsbinaryquantizerparams) | configure binary quantizer, e.g. threshold |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix |
| `quantizer` | out | [`cuvsBinaryQuantizer_t`](/api-reference/c-api-preprocessing-quantize-binary#cuvsbinaryquantizer) | trained binary quantizer |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbinaryquantizertransform"></a>
### cuvsBinaryQuantizerTransform

Applies binary quantization transform to the given dataset

```c
cuvsError_t cuvsBinaryQuantizerTransform(cuvsResources_t res,
DLManagedTensor* dataset,
DLManagedTensor* out);
```

This applies binary quantization to a dataset, changing any positive values to a bitwise 1. This is useful for searching with the BitwiseHamming distance type.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix to transform |
| `out` | out | `DLManagedTensor*` | a row-major host or device matrix to store transformed data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbinaryquantizertransformwithparams"></a>
### cuvsBinaryQuantizerTransformWithParams

Applies binary quantization transform to the given dataset

```c
cuvsError_t cuvsBinaryQuantizerTransformWithParams(cuvsResources_t res,
cuvsBinaryQuantizer_t quantizer,
DLManagedTensor* dataset,
DLManagedTensor* out);
```

This applies binary quantization to a dataset, changing any values that are larger than the threshold specified in the param to a bitwise 1. This is useful for searching with the BitwiseHamming distance type.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `quantizer` | in | [`cuvsBinaryQuantizer_t`](/api-reference/c-api-preprocessing-quantize-binary#cuvsbinaryquantizer) | binary quantizer |
| `dataset` | in | `DLManagedTensor*` | a row-major host or device matrix to transform |
| `out` | out | `DLManagedTensor*` | a row-major host or device matrix to store transformed data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
