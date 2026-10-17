---
slug: api-reference/cpp-api-core-dataset-sq
---

# SQ

_Source header: `cuvs/core/dataset/sq.hpp`_

## Types

<a id="sq-spec"></a>
### sq_spec

Scalar-quantized dataset specification: every component of a vector becomes one code, so the

dictionary is only the range the codes are mapped back onto and the dimension is still a property of the data.

```cpp
template <typename MathT, typename CodePolicy>
struct sq_spec;
```
