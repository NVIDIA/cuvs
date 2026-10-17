---
slug: api-reference/cpp-api-core-dataset-vpq
---

# Vpq

_Source header: `cuvs/core/dataset/vpq.hpp`_

## Types

<a id="vpq-spec"></a>
### vpq_spec

VQ+PQ compressed dataset specification, mirroring the VPQ dataset of cuvs/neighbors/common.hpp:

the two codebooks form the dictionary, while the data slot keeps the encoded rows - the VQ label in the row prefix, followed by the packed PQ codes.

`BookPolicy` is the container policy of the codebooks (elements of `MathT`) and `CodePolicy` the one of the encoded rows (elements of `uint8_t`); a bound policy cannot be rebound to another element type, hence the two parameters.

```cpp
template <typename MathT, typename BookPolicy, typename CodePolicy>
struct vpq_spec;
```
