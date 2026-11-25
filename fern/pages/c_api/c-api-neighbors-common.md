---
slug: api-reference/c-api-neighbors-common
---

# Common

_Source header: `cuvs/neighbors/common.h`_

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
