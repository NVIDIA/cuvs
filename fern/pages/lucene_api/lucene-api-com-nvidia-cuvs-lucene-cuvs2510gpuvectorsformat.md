---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvs2510gpuvectorsformat
---

# CuVS2510GPUVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class CuVS2510GPUVectorsFormat extends KnnVectorsFormat
```

Extends upon the KnnVectorsFormat - Encodes/decodes per-document vector and any associated indexing structures required to support
GPU-based accelerated nearest-neighbor search.

## Public Members

### CuVS2510GPUVectorsFormat

```java
public CuVS2510GPUVectorsFormat()
```

Initializes the `CuVS2510GPUVectorsFormat` with default parameter values.

**Throws**

| Type | Description |
| --- | --- |
| `LibraryException` | if the native library fails to load |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.java:55`_

### CuVS2510GPUVectorsFormat

```java
public CuVS2510GPUVectorsFormat(GPUSearchParams gpuSearchParams)
```

Initializes the `CuVS2510GPUVectorsFormat` with an instance of `GPUSearchParams`.

**Parameters**

| Name | Description |
| --- | --- |
| `gpuSearchParams` | An instance of `GPUSearchParams` |

**Throws**

| Type | Description |
| --- | --- |
| `LibraryException` | if the native library fails to load |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.java:65`_

### CuVS2510GPUVectorsFormat

```java
public CuVS2510GPUVectorsFormat( GPUSearchParams gpuSearchParams, FilterBitsetCacheConfig filterCacheConfig)
```

Initializes the format with GPU search and filter-bitset-cache parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `gpuSearchParams` | GPU index and search parameters |
| `filterCacheConfig` | filter-bitset-cache configuration |

**Throws**

| Type | Description |
| --- | --- |
| `LibraryException` | if the native library fails to load |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.java:76`_

### fieldsWriter

```java
@Override public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException
```

Returns a KnnVectorsReader instance to write the vectors to the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.java:86`_

### fieldsReader

```java
@Override public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException
```

Returns a KnnVectorsReader instance to read the vectors from the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.java:96`_

### getMaxDimensions

```java
@Override public int getMaxDimensions(String fieldName)
```

Returns the maximum number of vector dimensions supported by this codec for the given field name.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.java:106`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsFormat.java:25`_
