---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswscalarquantizedvectorsformat
---

# LuceneAcceleratedHNSWScalarQuantizedVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWScalarQuantizedVectorsFormat extends KnnVectorsFormat
```

cuVS based Scalar Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.

## Public Members

### LuceneAcceleratedHNSWScalarQuantizedVectorsFormat

```java
public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat()
```

Initializes `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat` with default values.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:60`_

### LuceneAcceleratedHNSWScalarQuantizedVectorsFormat

```java
public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat( AcceleratedHNSWParams acceleratedHNSWParams)
```

Initializes `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat` with the given threads, graph degree, etc.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | An instance of `AcceleratedHNSWParams` |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:69`_

### fieldsWriter

```java
@Override public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException
```

Returns a KnnVectorsWriter to write the scalar quantized vectors to the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:78`_

### fieldsReader

```java
@Override public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException
```

Returns a KnnVectorsReader to read the scalar quantized vectors from the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:105`_

### getMaxDimensions

```java
@Override public int getMaxDimensions(String fieldName)
```

Returns the maximum number of vector dimensions supported by this Codec for the given field name.

Returns 4096 when cuVS is supported for the current thread. Otherwise, returns `KnnVectorsFormat#DEFAULT_MAX_DIMENSIONS`, which is 1024 in the targeted Lucene version, for the
CPU fallback.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:123`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:23`_
