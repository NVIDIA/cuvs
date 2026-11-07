---
slug: api-reference/c-api-preprocessing-quantize-bbq
---

# Bbq

_Source header: `cuvs/preprocessing/quantize/bbq.h`_

## C API for Better Binary Quantization datasets

<a id="cuvsbbqcodelayout-t"></a>
### cuvsBbqCodeLayout_t

Storage layout of BBQ/OSQ quantized component codes in each dataset row.

```c
typedef enum {
  CUVS_BBQ_CODE_LAYOUT_PACKED_1B = 0,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_4B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_7B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_8B
} cuvsBbqCodeLayout_t;
```

**Values**

| Name | Value | Description |
| --- | --- | --- |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_1B` | `0` | Each dimension is quantized to a single bit and packed into bytes. Reflects Lucene's OptimizedScalarQuantizer.packAsBinary. |
| `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B` | `` | Each dimension is quantized to 2 bits, stored as 2 bitplanes. Reflects Lucene's OptimizedScalarQuantizer.transposeDibit. SIMT popc path only (paired with a transposed_4b or packed_1b operand); |
| `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B` | `` | Each dimension is quantized to 4 bits, optimized for bitwise operations. Reflects Lucene's OptimizedScalarQuantizer.transposeHalfByte. the first bit of every dimension is in the first set dimensions bits, or (dimensions/8) bytes. The second, third, and fourth bits are in the second, third, and fourth set of dimensions bits, respectively. Format used for queries. |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_4B` | `` | Each dimension is quantized to 4 bits, two values are packed into each output byte. |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_7B` | `` | Each dimension is quantized to 7 bits and treated as a signed value. |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_8B` | `` | Each dimension is quantized to 8 bits and treated as an unsigned value. |

<a id="cuvsbbqquantizer"></a>
### cuvsBbqQuantizer

Better Binary Quantization ([BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch)) is a vector-quantization approach used in Elasticsearch and Apache Lucene. It builds on ideas introduced in RaBitQ([Gao and Long](https://arxiv.org/pdf/2405.12497, [Gao et al.](https://arxiv.org/pdf/2409.09913)): residual binary codes around a centroid, corrective factors, and efficient bitwise comparison of codes at different bit widths. Lucene implements this as optimized scalar quantization (OSQ) with packed and bit-plane layouts; Elasticsearch exposes it as BBQ.

BBQ in cuVS designed to be compatible with the Lucene/Elasticsearch dataset: a single shared centroid, no random rotation, and OSQ codes.

RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain additional per-vector information so search is better than naïve sign-bit comparison. They differ in transformation and scale representation. RaBitQ commonly separates residual magnitude from direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector scalar intervals to interpret the compressed residual codes.

```c
typedef struct cuvsBbqQuantizer {
  uintptr_t addr;
  DLDataType dtype;
  bool is_owning;
} cuvsBbqQuantizer;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |
| `is_owning` | `bool` |  |

<a id="cuvsbbqquantizercreateview"></a>
### cuvsBbqQuantizerCreateView

Create a BBQ quantizer view from caller-owned device tensors.

```c
cuvsError_t cuvsBbqQuantizerCreateView(
DLManagedTensor* codes,
DLManagedTensor* lower_intervals,
DLManagedTensor* upper_intervals,
DLManagedTensor* additional_corrections,
DLManagedTensor* quantized_component_sums,
DLManagedTensor* centroid,
DLManagedTensor* dequant_delta,
DLManagedTensor* dequant_sum_delta,
DLManagedTensor* row_norm,
cuvsBbqCodeLayout_t layout,
cuvsDistanceType metric,
float centroid_norm_sq,
cuvsBbqQuantizer_t* quantizer);
```

Tensors are not copied and must remain valid while a derived dataset is in use. `centroid_norm_sq` is copied to device memory owned by the quantizer handle, and datasets and indexes derived from the handle refer to that copy. The handle must therefore not be destroyed while any dataset or index derived from it is in use.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `codes` | in | `DLManagedTensor*` | uint8 device matrix containing encoded rows |
| `lower_intervals` | in | `DLManagedTensor*` | float32 device vector with one lower interval per row |
| `upper_intervals` | in | `DLManagedTensor*` | float32 device vector with one upper interval per row |
| `additional_corrections` | in | `DLManagedTensor*` | float32 device vector with one correction per row |
| `quantized_component_sums` | in | `DLManagedTensor*` | int32 device vector with one component sum per row |
| `centroid` | in | `DLManagedTensor*` | device vector containing the dataset centroid |
| `dequant_delta` | in | `DLManagedTensor*` | float32 device vector with one dequantization delta per row |
| `dequant_sum_delta` | in | `DLManagedTensor*` | float32 device vector with one delta-times-sum value per row |
| `row_norm` | in | `DLManagedTensor*` | float32 device vector with one original-space squared norm per row |
| `layout` | in | [`cuvsBbqCodeLayout_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqcodelayout-t) | encoded code layout |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | distance metric associated with the encoded dataset |
| `centroid_norm_sq` | in | `float` | squared norm of the centroid |
| `quantizer` | out | [`cuvsBbqQuantizer_t*`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | newly allocated non-owning quantizer handle |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizerdestroy"></a>
### cuvsBbqQuantizerDestroy

Destroy a BBQ quantizer without destroying its caller-owned tensors.

```c
cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer);
```

Frees the quantizer's device copy of `centroid_norm_sq`. Datasets and indexes derived from the quantizer must be destroyed first.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` | in | [`cuvsBbqQuantizer_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | quantizer handle to destroy |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsdatasetmakebbqview"></a>
### cuvsDatasetMakeBbqView

Create a non-owning device BBQ dataset view.

```c
cuvsError_t cuvsDatasetMakeBbqView(cuvsResources_t res,
cuvsBbqQuantizer_t* quantizers,
size_t num_quantizers,
cuvsDataset_t* dataset);
```

Accepts one symmetric quantizer or two compatible asymmetric quantizers. Both asymmetric quantizers must encode the same rows, in the same order, against the same centroid. The dataset, and any index built from it, refers to the quantizers' data, so the quantizer handles must outlive them.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuVS resources |
| `quantizers` | in | [`cuvsBbqQuantizer_t*`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | array containing one or two BBQ quantizer handles |
| `num_quantizers` | in | `size_t` | number of elements in `quantizers` |
| `dataset` | out | `cuvsDataset_t*` | newly allocated non-owning BBQ dataset handle |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizerparams"></a>
### cuvsBbqQuantizerParams

Parameters for encoding a dataset with `cuvsBbqQuantizerTransform`.

```c
struct cuvsBbqQuantizerParams {
  cuvsBbqCodeLayout_t doc_layout;
  cuvsBbqCodeLayout_t query_layout;
  cuvsDistanceType metric;
  float lambda;
  int iters;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `doc_layout` | [`cuvsBbqCodeLayout_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqcodelayout-t) | Code layout of the encoded rows on the document side of a distance. |
| `query_layout` | [`cuvsBbqCodeLayout_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqcodelayout-t) | Code layout of the encoded rows on the query side of a distance. When it differs from `doc_layout`, the dataset is encoded a second time in this layout. |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | Selects the per-row additional correction: \|\|x - c\|\|^2 for L2, x.c otherwise. |
| `lambda` | `float` | Relative interval weight of the quantization error against the norm-preserving term of the loss. |
| `iters` | `int` | Coordinate descent iterations per row. |

<a id="cuvsbbqquantizerparamscreate"></a>
### cuvsBbqQuantizerParamsCreate

Allocate BBQ quantizer params, and populate with default values

```c
cuvsError_t cuvsBbqQuantizerParamsCreate(cuvsBbqQuantizerParams_t* params);
```

Defaults: `CUVS_BBQ_CODE_LAYOUT_PACKED_1B` documents, `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B` queries, `L2Expanded`, `lambda` 0.1 and 5 `iters`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | out | [`cuvsBbqQuantizerParams_t*`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizerparams) | cuvsBbqQuantizerParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizerparamsdestroy"></a>
### cuvsBbqQuantizerParamsDestroy

De-allocate BBQ quantizer params

```c
cuvsError_t cuvsBbqQuantizerParamsDestroy(cuvsBbqQuantizerParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsBbqQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizerparams) | cuvsBbqQuantizerParams_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizerbuild"></a>
### cuvsBbqQuantizerBuild

Build a BBQ quantizer from a dataset: computes the centroid shared by every row.

```c
cuvsError_t cuvsBbqQuantizerBuild(cuvsResources_t res,
DLManagedTensor* dataset,
cuvsBbqQuantizer_t* quantizer);
```

The returned handle owns the centroid and its squared norm, and only serves to encode datasets with `cuvsBbqQuantizerTransform`: it holds no encoded rows, so `cuvsDatasetMakeBbqView` rejects it. Destroy it with `cuvsBbqQuantizerDestroy`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuVS resources |
| `dataset` | in | `DLManagedTensor*` | float32 row-major device matrix |
| `quantizer` | out | [`cuvsBbqQuantizer_t*`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | newly allocated owning quantizer handle |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizertransform"></a>
### cuvsBbqQuantizerTransform

Encode a dataset against the centroid of a BBQ quantizer into an owning BBQ dataset.

```c
cuvsError_t cuvsBbqQuantizerTransform(cuvsResources_t res,
cuvsBbqQuantizerParams_t params,
cuvsBbqQuantizer_t quantizer,
DLManagedTensor* in_dataset,
cuvsDataset_t* bbq_dataset);
```

The dataset holds one storage in `params->doc_layout`, plus one in `params->query_layout` when it differs, and its own copy of the centroid, so it does not depend on `quantizer` or `dataset` once returned. `quantizer` is either built by `cuvsBbqQuantizerBuild`, or a float32 view from `cuvsBbqQuantizerCreateView`, whose centroid is then reused. Destroy the result with `cuvsDatasetDestroy`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuVS resources |
| `params` | in | [`cuvsBbqQuantizerParams_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizerparams) | encoding parameters |
| `quantizer` | in | [`cuvsBbqQuantizer_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqquantizer) | BBQ quantizer whose centroid the rows are encoded against |
| `in_dataset` | in | `DLManagedTensor*` | float32 row-major device matrix with the quantizer's dimension |
| `bbq_dataset` | out | `cuvsDataset_t*` | newly allocated owning `CUVS_DATASET_LAYOUT_BBQ` device dataset handle |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
