---
slug: api-reference/c-api-preprocessing-quantize-bbq
---

# Bbq

_Source header: `cuvs/preprocessing/quantize/bbq.h`_

## C API for Better Binary Quantization datasets

<a id="cuvsbbqcodelayout-t"></a>
### cuvsBbqCodeLayout_t

Storage layout of the quantized component codes in each dataset row.

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

| Name | Value |
| --- | --- |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_1B` | `0` |
| `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B` | `` |
| `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B` | `` |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_4B` | `` |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_7B` | `` |
| `CUVS_BBQ_CODE_LAYOUT_PACKED_8B` | `` |

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

Tensors are not copied and must remain valid while a derived dataset is in use.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `codes` |  | `DLManagedTensor*` |  |
| `lower_intervals` |  | `DLManagedTensor*` |  |
| `upper_intervals` |  | `DLManagedTensor*` |  |
| `additional_corrections` |  | `DLManagedTensor*` |  |
| `quantized_component_sums` |  | `DLManagedTensor*` |  |
| `centroid` |  | `DLManagedTensor*` |  |
| `dequant_delta` |  | `DLManagedTensor*` |  |
| `dequant_sum_delta` |  | `DLManagedTensor*` |  |
| `row_norm` |  | `DLManagedTensor*` |  |
| `layout` |  | [`cuvsBbqCodeLayout_t`](/api-reference/c-api-preprocessing-quantize-bbq#cuvsbbqcodelayout-t) |  |
| `metric` |  | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) |  |
| `centroid_norm_sq` |  | `float` |  |
| `quantizer` |  | `cuvsBbqQuantizer_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsbbqquantizerdestroy"></a>
### cuvsBbqQuantizerDestroy

Destroy a BBQ quantizer without destroying its caller-owned tensors.

```c
cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `quantizer` |  | `cuvsBbqQuantizer_t` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
