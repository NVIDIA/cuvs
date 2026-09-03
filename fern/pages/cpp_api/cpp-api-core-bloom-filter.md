---
slug: api-reference/cpp-api-core-bloom-filter
---

# Bloom Filter

_Source header: `cuvs/core/bloom_filter.hpp`_

## Types

<a id="core-bloom-filter"></a>
### core::bloom_filter

cuVS-owned Bloom filter wrapper with opaque implementation.

This class intentionally hides cuCollections types from the cuVS public API. The wrapper supports the expected bulk host APIs used by ANN workflows.

```cpp
class bloom_filter;
```
