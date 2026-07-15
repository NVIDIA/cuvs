---
slug: api-reference/c-api-neighbors-mg-common
---

# Multi-GPU Common

_Source header: `cuvs/neighbors/mg_common.h`_

## Multi-GPU common types and enums

<a id="cuvsmultigpudistributionmode"></a>
### cuvsMultiGpuDistributionMode

Distribution mode for multi-GPU indexes

```c
typedef enum;
```

<a id="cuvsmultigpureplicatedsearchmode"></a>
### cuvsMultiGpuReplicatedSearchMode

Search mode when using a replicated index

```c
typedef enum {
  CUVS_NEIGHBORS_MG_LOAD_BALANCER = 0,
  CUVS_NEIGHBORS_MG_ROUND_ROBIN = 1
} cuvsMultiGpuReplicatedSearchMode;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_NEIGHBORS_MG_LOAD_BALANCER` | `0` |
| `CUVS_NEIGHBORS_MG_ROUND_ROBIN` | `1` |

<a id="cuvsmultigpushardedmergemode"></a>
### cuvsMultiGpuShardedMergeMode

Merge mode when using a sharded index

```c
typedef enum {
  CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK = 0,
  CUVS_NEIGHBORS_MG_TREE_MERGE = 1
} cuvsMultiGpuShardedMergeMode;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK` | `0` |
| `CUVS_NEIGHBORS_MG_TREE_MERGE` | `1` |
