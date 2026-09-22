---
slug: api-reference/cpp-api-core-roaring-allowlist
---

# Roaring Allowlist

_Source header: `cuvs/core/roaring_allowlist.hpp`_

## Types

<a id="core-roaring-allowlist-view"></a>
### core::roaring_allowlist_view

Non-owning device view of one immutable Roaring allowlist.

The view contains an opaque pointer to an already initialized device-side cuCollections reference plus immutable shape and cardinality metadata. Creating or copying it is O(1) and performs no allocation, parsing, kernel launch, or synchronization. The owning

```cpp
class roaring_allowlist_view;
```

<a id="core-roaring-allowlist"></a>
### core::roaring_allowlist

Owning immutable exact Roaring allowlist over CAGRA dataset-row IDs.

Build an allowlist through `from_ids`, then pass its zero-copy `view` to a cuvs::neighbors::filtering::roaring_bitmap_filter. A filter maps one such view to each query; owners remain independent and can therefore be reused across filters and queries.

Construction sorts IDs on the GPU unless `pre_sorted` is true. Setting `pre_sorted` promises that IDs are already in strictly increasing order; this promise is not verified. IDs must be unique and smaller than `dataset_rows`. The encoded bytes and the initialized cuco::experimental::roaring_bitmap_ref&lt;uint32_t&gt; are retained on the device. Creating a view never copies or reparses them, and CAGRA search performs no Roaring initialization.

ID-based construction emits the standard portable 32-bit Roaring array and bitmap container forms. Each ID is partitioned by its high 16 bits; the low 16 bits are stored as an array for at most 4,096 values in a partition and as an 8 KiB bitmap otherwise.

```cpp
class roaring_allowlist;
```
