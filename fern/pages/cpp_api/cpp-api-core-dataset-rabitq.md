---
slug: api-reference/cpp-api-core-dataset-rabitq
---

# Rabitq

_Source header: `cuvs/core/dataset/rabitq.hpp`_

## Types

<a id="rabitq-spec"></a>
### rabitq_spec

RaBitQ-quantized dataset specification.

The data holds everything that scales with the number of rows:

- `codes` [n_rows, padded_dim]: one `CodeT` per padded dimension, each holding an unsigned `ex_bits + 1` bit value;
- `factors` [n_rows, 3]: the per-row `(f_add, f_rescale, f_error)` triplet. Both are what the full-factor `preprocessing::quantize::rabitq::transform()` writes when it is called without a centroid.

The manifest is the RaBitQ quantizer (rotator and encoder) that produced them. It also knows the logical dimension `dim`, which the codes cannot tell: their width is padded to a multiple of 32.

```cpp
template <typename MathT, typename CodeT, typename CodePolicy, typename FactorPolicy>
struct rabitq_spec;
```
