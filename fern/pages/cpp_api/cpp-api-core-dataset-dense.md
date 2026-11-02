---
slug: api-reference/cpp-api-core-dataset-dense
---

# Dense

_Source header: `cuvs/core/dataset/dense.hpp`_

## Types

<a id="mdarray-spec"></a>
### mdarray_spec

Dense plain/padded dataset implemented via raft::mdarray.

```cpp
template <typename LayoutPolicy, typename ContainerPolicy>
struct mdarray_spec;
```
