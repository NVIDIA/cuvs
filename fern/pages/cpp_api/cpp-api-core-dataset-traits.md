---
slug: api-reference/cpp-api-core-dataset-traits
---

# Traits

_Source header: `cuvs/core/dataset/traits.hpp`_

## Types

<a id="owning-dataset-for-view"></a>
### owning_dataset_for_view

Maps a dataset view type to its owning (allocating) dataset counterpart. Used by serialization to

type the out_dataset output parameter; a kind that is not a plain spec-based dataset adds a specialization.

```cpp
template <typename DatasetViewT>
struct owning_dataset_for_view;
```
