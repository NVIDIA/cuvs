---
slug: api-reference/cpp-api-core-dataset-view
---

# View

_Source header: `cuvs/core/dataset/view.hpp`_

## Types

<a id="dataset-view"></a>
### dataset_view

A non-owning view of a dataset: a view of the data and, for a compound dataset, a view of the

manifest that gives it meaning. Cheap to copy and to pass by value, like an mdspan.

The dataset consists of two components: the data and the manifest.

The data member is the main storage of the dataset: it contains the raw (maybe encoded/compressed) data. The data grows linearly with the number of rows in the dataset. Examples of the data content: floating point vectors, encoded byte representations, per-point labels scaling factors, cluster ids, etc - everything that is associated with every data row.

The manifest member is an optional component that contains everything to give meaning to the data. Its size is bounded and doesn't depend directly on the number of rows in the dataset. Examples of the manifest content: codebooks, quantization ranges, global scaling factors, extra data transforms (such as rotation matrices) etc. - everything that is relevant for the entire dataset, not just individual rows.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset_view {
  static constexpr bool is_const_view;
  static constexpr bool is_compound;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `is_const_view` | `static constexpr bool` |  |
| `is_compound` | `static constexpr bool` |  |
