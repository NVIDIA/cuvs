---
slug: api-reference/cpp-api-core-dataset-view
---

# View

_Source header: `cuvs/core/dataset/view.hpp`_

## Types

<a id="dataset-view"></a>
### dataset_view

A non-owning view of a dataset: a view of the data and, for a compressed dataset, a view of the

dictionary that gives it meaning. Cheap to copy and to pass by value, like an mdspan.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset_view {
  static constexpr bool is_const_view;
  static constexpr bool is_compressed;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `is_const_view` | `static constexpr bool` |  |
| `is_compressed` | `static constexpr bool` |  |
