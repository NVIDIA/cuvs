---
slug: api-reference/cpp-api-preprocessing-quantize-bbq
---

# Bbq

_Source header: `cuvs/preprocessing/quantize/bbq.hpp`_

## Better Binary Quantization utilities

<a id="preprocessing-quantize-bbq-bbq-code-layout"></a>
### preprocessing::quantize::bbq::bbq_code_layout

Layout of BBQ quantized codes in each dataset row.

```cpp
enum class bbq_code_layout {
  single_bit,
  dibit,
  transpose_half_byte,
  packed_nibble,
  seven_bit,
  unsigned_byte
};
```

**Values**

| Name | Value |
| --- | --- |
| `single_bit` | `` |
| `dibit` | `` |
| `transpose_half_byte` | `` |
| `packed_nibble` | `` |
| `seven_bit` | `` |
| `unsigned_byte` | `` |

<a id="preprocessing-quantize-bbq-bbq-quantizer"></a>
### preprocessing::quantize::bbq::bbq_quantizer

Owning structure for BBQ quantizer data.

```cpp
template <typename DataT, typename IdxT, typename Accessor>
struct bbq_quantizer {
  dense_owning_matrix<uint8_t, IdxT, Accessor> codes;
  dense_owning_vector<float, IdxT, Accessor> lower_intervals;
  dense_owning_vector<float, IdxT, Accessor> upper_intervals;
  dense_owning_vector<float, IdxT, Accessor> additional_corrections;
  dense_owning_vector<int32_t, IdxT, Accessor> quantized_component_sums;
  dense_owning_vector<DataT, IdxT, Accessor> centroid;
  uint32_t bits;
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
  float centroid_norm_sq;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `codes` | `dense_owning_matrix<uint8_t, IdxT, Accessor>` |  |
| `lower_intervals` | `dense_owning_vector<float, IdxT, Accessor>` |  |
| `upper_intervals` | `dense_owning_vector<float, IdxT, Accessor>` |  |
| `additional_corrections` | `dense_owning_vector<float, IdxT, Accessor>` |  |
| `quantized_component_sums` | `dense_owning_vector<int32_t, IdxT, Accessor>` |  |
| `centroid` | `dense_owning_vector<DataT, IdxT, Accessor>` |  |
| `bits` | `uint32_t` |  |
| `layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `centroid_norm_sq` | `float` |  |

<a id="preprocessing-quantize-bbq-bbq-quantizer-view"></a>
### preprocessing::quantize::bbq::bbq_quantizer_view

View structure for BBQ quantizer data.

```cpp
template <typename DataT, typename IdxT, typename Accessor>
struct bbq_quantizer_view {
  dense_view_matrix<const uint8_t, IdxT, Accessor> codes;
  dense_view_vector<const float, IdxT, Accessor> lower_intervals;
  dense_view_vector<const float, IdxT, Accessor> upper_intervals;
  dense_view_vector<const float, IdxT, Accessor> additional_corrections;
  dense_view_vector<const int32_t, IdxT, Accessor> quantized_component_sums;
  dense_view_vector<const DataT, IdxT, Accessor> centroid;
  uint32_t bits;
  bbq_code_layout layout;
  cuvs::distance::DistanceType metric;
  float centroid_norm_sq;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `codes` | `dense_view_matrix<const uint8_t, IdxT, Accessor>` |  |
| `lower_intervals` | `dense_view_vector<const float, IdxT, Accessor>` |  |
| `upper_intervals` | `dense_view_vector<const float, IdxT, Accessor>` |  |
| `additional_corrections` | `dense_view_vector<const float, IdxT, Accessor>` |  |
| `quantized_component_sums` | `dense_view_vector<const int32_t, IdxT, Accessor>` |  |
| `centroid` | `dense_view_vector<const DataT, IdxT, Accessor>` |  |
| `bits` | `uint32_t` |  |
| `layout` | [`bbq_code_layout`](/api-reference/cpp-api-preprocessing-quantize-bbq#preprocessing-quantize-bbq-bbq-code-layout) |  |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `centroid_norm_sq` | `float` |  |
