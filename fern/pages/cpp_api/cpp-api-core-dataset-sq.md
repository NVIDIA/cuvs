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
template <typename MathT,
typename CodePolicy,
typename QuantizerT = cuvs::preprocessing::quantize::scalar::quantizer<MathT>>
struct sq_spec;
```

<a id="device-per-dim-sq-dataset"></a>
### device_per_dim_sq_dataset

Owns row-major INT8 codes and their per-dimension offsets/shared scale.

```cpp
using device_per_dim_sq_dataset =
sq_dataset<float,
int64_t,
float,
raft::device_accessor<raft::device_container_policy<int8_t>>,
cuvs::preprocessing::quantize::scalar::per_dim_quantizer>;
```
