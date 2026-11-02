---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-filterbitsetcacheconfig
---

# FilterBitsetCacheConfig

_Java package: `com.nvidia.cuvs.lucene`_

```java
public record FilterBitsetCacheConfig(boolean enabled, long maxBytes)
```

Configuration for the filter-bitset cache used by multi-segment GPU searches.

This configuration is local to one vectors-format instance. It does not configure cuvs-java's
process-wide filter-bitset device pool.

The byte budget is a retention cap, not a preallocation. Packed host arrays and their
device-side mirrors are allocated as cache entries are populated. Actual host usage can exceed
the accounted payload bytes because of object overhead, temporary filter construction, and
in-flight handles.

**Parameters**

| Name | Description |
| --- | --- |
| `enabled` | whether filter bitsets are cached between searches |
| `maxBytes` | maximum bytes cached by this format; must be positive when caching is enabled |

## Public Members

### FilterBitsetCacheConfig

```java
public static final FilterBitsetCacheConfig DEFAULT = new FilterBitsetCacheConfig(true, DEFAULT_MAX_BYTES)
```

Default configuration used by SPI-created codecs and vector formats.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/FilterBitsetCacheConfig.java:27`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/FilterBitsetCacheConfig.java:22`_
