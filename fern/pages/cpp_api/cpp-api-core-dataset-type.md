---
slug: api-reference/cpp-api-core-dataset-type
---

# Type

_Source header: `cuvs/core/dataset/type.hpp`_

## Types

<a id="dataset"></a>
### dataset

An owning dataset: the data and, for a compound dataset, the manifest that gives it meaning.

Ownership is unique, so the type is move-only whenever the underlying storage is.

See the dataset_view documentation for more details on the dataset's components.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset {
  using manifest_type;
  static constexpr bool is_compound;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `manifest_type` | `using` | Optional slot that contains everything to give meaning to the encoded data; `std::monostate` for a plain dataset. |
| `is_compound` | `static constexpr bool` |  |

<a id="manifest-type"></a>
### manifest_type

Optional slot that contains everything to give meaning to the encoded data;

```cpp
using manifest_type            = typename view_type::manifest_type;
```
