---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-gpuknnfloatvectorquery
---

# GPUKnnFloatVectorQuery

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class GPUKnnFloatVectorQuery extends KnnFloatVectorQuery
```

Extends `KnnFloatVectorQuery` for GPU-only search.

When all index segments use `CuVS2510GPUVectorsReader`, `#rewrite` delegates a
single multi-partition search to cuVS, passing one Lucene segment per cuVS partition. cuVS
runs the per-partition CAGRA searches, applies distance post-processing, and performs the
cross-partition top-k merge internally; the returned arrays are mapped to Lucene doc IDs on
the host. The effective CAGRA algorithm (SINGLE_CTA or MULTI_KERNEL) is selected by cuVS
based on `searchAlgo` and `itopk_size`, with MULTI_KERNEL handling k beyond
SINGLE_CTA's per-partition cap.

If the query has an explicit `filter`, or if any segment carries live-document deletes,
the acceptance mask (filter ∩ liveDocs) is packed into one `FilterBitsetHandle` per segment
and passed as that partition's filter. The host-side packed arrays are cached per unique
(filter, single-segment reader key, field) triple via `FilterBitsetCache`; the device
upload is cached inside the handle itself across threads.

Falls back to the standard per-segment Lucene path when the optimized path cannot be
applied: mixed segment types, a missing CAGRA index for the field on any segment, or segments
whose built CAGRA graphs differ in degree (a single multi-partition request requires a uniform
graph degree, and a small segment can have its degree truncated at build time).

## Public Members

### GPUKnnFloatVectorQuery

```java
public GPUKnnFloatVectorQuery( String field, float[] target, int k, Query filter, int iTopK, int searchWidth)
```

Initializes `GPUKnnFloatVectorQuery` with `CagraSearchParams.SearchAlgo#AUTO`,
and max_iterations auto-selected (0).

**Parameters**

| Name | Description |
| --- | --- |
| `field` | the vector field name |
| `target` | the query vector |
| `k` | the number of nearest neighbors to return |
| `filter` | optional pre-filter query |
| `iTopK` | CAGRA itopk_size parameter |
| `searchWidth` | CAGRA search_width parameter |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUKnnFloatVectorQuery.java:90`_

### GPUKnnFloatVectorQuery

```java
public GPUKnnFloatVectorQuery( String field, float[] target, int k, Query filter, int iTopK, int searchWidth, int threadBlockSize, int maxIterations, CagraSearchParams.SearchAlgo searchAlgo)
```

Initializes `GPUKnnFloatVectorQuery`.

**Parameters**

| Name | Description |
| --- | --- |
| `field` | the vector field name |
| `target` | the query vector |
| `k` | the number of nearest neighbors to return |
| `filter` | optional pre-filter query |
| `iTopK` | CAGRA itopk_size parameter |
| `searchWidth` | CAGRA search_width parameter |
| `threadBlockSize` | CAGRA thread_block_size (0 = auto) |
| `maxIterations` | CAGRA max_iterations (0 = auto) |
| `searchAlgo` | CAGRA search algorithm |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUKnnFloatVectorQuery.java:108`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUKnnFloatVectorQuery.java:72`_
