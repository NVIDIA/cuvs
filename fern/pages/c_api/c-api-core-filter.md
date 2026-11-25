---
slug: api-reference/c-api-core-filter
---

# Filter

_Source header: `cuvs/core/filter.h`_

## Filters APIs

<a id="cuvsfiltertype"></a>
### cuvsFilterType

Enum to denote filter type.

```c
enum cuvsFilterType {
  NO_FILTER = 0,
  BITSET = 1,
  BITMAP = 2
};
```

**Values**

| Name | Value |
| --- | --- |
| `NO_FILTER` | `0` |
| `BITSET` | `1` |
| `BITMAP` | `2` |

<a id="cuvsfilter"></a>
### cuvsFilter

Struct to hold a filter address and its type.

The meaning of `addr` depends on `type`:

- `NO_FILTER`: `addr` is unused (set to 0).
- `BITSET`:    `addr` is the address of a `DLManagedTensor*` holding the bitset on device. A set bit means the corresponding row survives the filter.

The filter object pointed to by `addr` is owned by the caller; the library performs no caching of the underlying bitset across search calls. Allocating and populating the device bitset may be more expensive than a single filtered search, so callers that issue repeated searches against the same filter (e.g. many queries over one index) should build the bitset once and reuse the same cuvsFilter across those calls rather than rebuild it per search. Reusing the bitset is essential for realizing the full throughput of filtered search.

```c
typedef struct {
  uintptr_t addr;
} cuvsFilter;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
