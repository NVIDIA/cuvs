---
slug: api-reference/cpp-api-core-dataset-type
---

# Type

_Source header: `cuvs/core/dataset/type.hpp`_

## Types

<a id="dataset"></a>
### dataset

An owning dataset: the data and, for a compressed dataset, the dictionary that gives it meaning.

Ownership is unique, so the type is move-only whenever the underlying storage is.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset {
  using dictionary_type;
  static constexpr bool is_compressed;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `dictionary_type` | `using` | Optional slot that contains everything to give meaning to the encoded data; `std::monostate` for an uncompressed dataset. |
| `is_compressed` | `static constexpr bool` |  |

<a id="dictionary-type"></a>
### dictionary_type

Optional slot that contains everything to give meaning to the encoded data;

```cpp
using dictionary_type            = typename view_type::dictionary_type;
```
