---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene104acceleratedhnswscalarquantizedvectorsformat
---

# Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat extends KnnVectorsFormat
```

cuVS based scalar-quantized KnnVectorsFormat for indexing on the GPU and searching on the CPU.

Stores the vectors with Lucene's `Lucene104ScalarQuantizedVectorsFormat`, quantized to
one byte per dimension (`UNSIGNED_BYTE`), which replaced `Lucene99ScalarQuantizedVectorsFormat` in Lucene 10.4. On the GPU, the graph is built over the
vectors quantized to 7 bits, as `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat` does,
so the stored encoding mostly affects how the graph is searched on the CPU. One exception: the
flat format normalizes the vectors of COSINE fields when it flushes them, so the graph of a
flushed segment is built from normalized vectors, and that of a merged segment from the vectors
as indexed. Without a GPU, Lucene builds the graph on the CPU, scoring with the stored encoding.

## Public and Protected Members

### Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

```java
public Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat()
```

Creates the format with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:29`_

### Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

```java
public Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat( AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the format.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:38`_

### fieldsWriter

```java
@Override public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException
```

Returns a KnnVectorsWriter to write the scalar quantized vectors to the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseAcceleratedHNSWScalarQuantizedVectorsFormat.java:65`_

### fieldsReader

```java
@Override public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException
```

Returns a KnnVectorsReader to read the scalar quantized vectors from the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseAcceleratedHNSWScalarQuantizedVectorsFormat.java:135`_

### getMaxDimensions

```java
@Override public int getMaxDimensions(String fieldName)
```

Returns the maximum number of vector dimensions supported by this format for the given field
name.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/BaseAcceleratedHNSWScalarQuantizedVectorsFormat.java:165`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:23`_
