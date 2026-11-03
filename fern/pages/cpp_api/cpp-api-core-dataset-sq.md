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

Train and encode a device FP32 corpus with the selected range mode.

```cpp
inline device_sq_dataset<float, int64_t, float> build(
raft::resources const& res, params params, raft::device_matrix_view<const float, int64_t> dataset);
```

Reuse result.const_manifest_view() with transform for query batches. For separate training/encoding, use train and transform.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `params` |  |
| `dataset` |  | `raft::device_matrix_view<const float, int64_t>` |  |

**Returns**

`inline device_sq_dataset<float, int64_t, float>`
