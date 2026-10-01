---
slug: api-reference/java-api-com-nvidia-cuvs-multipartitioncagrasearch
---

# MultiPartitionCagraSearch

_Java package: `com.nvidia.cuvs`_

```java
public class MultiPartitionCagraSearch
```

Performs an approximate nearest neighbor search across multiple CAGRA index partitions in a
single native call. The caller supplies one `CagraQuery` whose query matrix is searched
against every partition; cuVS performs the per-partition searches, the cross-partition top-k
merge, and the post-processing internally, then returns the merged results.

As with `CagraIndex#search(CagraQuery)`, the query vectors may be either host- or
device-resident; host-resident query matrices are uploaded to the device internally.

## Public Members

### search

```java
public static MultiPartitionSearchResults search( CuVSResources resources, List<CagraIndex> indices, CagraQuery query, int k) throws Throwable
```

Searches multiple CAGRA index partitions for the global top-k nearest neighbors.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | shared `CuVSResources` handle |
| `indices` | one `CagraIndex` per partition, in partition order |
| `query` | a single `CagraQuery` whose query matrix is searched against every partition; its search parameters are shared across all partitions |
| `k` | number of global nearest neighbors to return per query |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/MultiPartitionCagraSearch.java:33`_

### search

```java
public static MultiPartitionSearchResults search( CuVSResources resources, List<CagraIndex> indices, CagraQuery query, int k, List<FilterBitsetHandle> filters) throws Throwable
```

Searches multiple CAGRA index partitions with optional per-partition device-side filters.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | shared `CuVSResources` handle |
| `indices` | one `CagraIndex` per partition, in partition order |
| `query` | a single `CagraQuery` whose query matrix is searched against every partition |
| `k` | number of global nearest neighbors to return per query |
| `filters` | one filter per partition, in the same order as `indices`, or `null`/empty for a fully unfiltered search. When non-null, its size must equal `indices.size()`; a `null` entry means no filter for that partition. Each handle must be obtained from `FilterBitsetHandle#create(long[])` for that partition's packed bitset; handles from other sources are not supported. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/MultiPartitionCagraSearch.java:53`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/MultiPartitionCagraSearch.java:21`_
