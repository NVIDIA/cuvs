---
slug: api-reference/cpp-api-neighbors-tiered-index
---

# Tiered Index

_Source header: `cuvs/neighbors/tiered_index.hpp`_

## Types

<a id="neighbors-tiered-index-index"></a>
### neighbors::tiered_index::index

Tiered Index class

```cpp
template <typename UpstreamT>
struct index : cuvs::neighbors::index {
  std::shared_ptr<detail::index_state<UpstreamT>> state;
  std::mutex write_mutex;
  mutable std::shared_mutex ann_mutex;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `state` | `std::shared_ptr<detail::index_state<UpstreamT>>` |  |
| `write_mutex` | `std::mutex` |  |
| `ann_mutex` | `mutable std::shared_mutex` |  |
