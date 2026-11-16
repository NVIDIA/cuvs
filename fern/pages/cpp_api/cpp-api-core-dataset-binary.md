---
slug: api-reference/cpp-api-core-dataset-binary
---

# Binary

_Source header: `cuvs/core/dataset/binary.hpp`_

## Types

<a id="binary-spec"></a>
### binary_spec

Binary-quantized dataset specification.

The data are bit-packed codes, `ceil(dim / 8)` bytes per row; bit `i` of a row is stored in bit `i % 8` of byte `i / 8`, and the unused bits of the last byte are zero. The manifest is the binary quantizer that produced the codes, so the logical dimension `dim` (the number of bits) never has to be guessed from the padded byte width: the quantizer knows it.

```cpp
template <typename MathT, typename CodePolicy, typename ThresholdPolicy>
struct binary_spec;
```
