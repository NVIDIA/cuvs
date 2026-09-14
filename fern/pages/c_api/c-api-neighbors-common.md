---
slug: api-reference/c-api-neighbors-common
---

# Common

_Source header: `cuvs/neighbors/common.h`_

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

Struct to hold address of cuvs::neighbors::prefilter and its type

`addr` points to a filter object owned by the caller; the library performs no caching of the underlying bitset across search calls. Allocating and populating the device bitset may be more expensive than a single filtered search, so callers that issue repeated searches against the same filter (e.g. many queries over one index) should build the bitset once and reuse the same cuvsFilter across those calls rather than rebuild it per search. Reusing the bitset is essential for realizing the full throughput of filtered search.

```c
typedef struct {
  uintptr_t addr;
} cuvsFilter;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |

## Index Merge

<a id="cuvsmergestrategy"></a>
### cuvsMergeStrategy

Strategy for merging indices.

```c
typedef enum {
  MERGE_STRATEGY_PHYSICAL = 0,
  MERGE_STRATEGY_LOGICAL = 1
} cuvsMergeStrategy;
```

**Values**

| Name | Value |
| --- | --- |
| `MERGE_STRATEGY_PHYSICAL` | `0` |
| `MERGE_STRATEGY_LOGICAL` | `1` |
