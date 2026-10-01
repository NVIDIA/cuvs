---
slug: api-reference/cpp-api-neighbors-composite-index
---

# Index

_Source header: `cuvs/neighbors/composite/index.hpp`_

## Types

<a id="neighbors-composite-composite-index"></a>
### neighbors::composite::composite_index

Composite index that searches multiple CAGRA sub-indices and merges results.

When the composite index contains multiple sub-indices, the user can set a stream pool in the input raft::resource to enable parallel search across sub-indices for improved performance.

Usage example:

```cpp
template <typename T, typename IdxT, typename OutputIdxT = IdxT>
class composite_index;
```
