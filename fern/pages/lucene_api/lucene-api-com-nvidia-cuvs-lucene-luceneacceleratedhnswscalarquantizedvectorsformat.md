---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswscalarquantizedvectorsformat
---

# LuceneAcceleratedHNSWScalarQuantizedVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWScalarQuantizedVectorsFormat extends KnnVectorsFormat
```

cuVS based Scalar Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.

Stores the vectors with Lucene's `Lucene99ScalarQuantizedVectorsFormat`, which Lucene
10.4 moved to its backward codecs. On Lucene 10.4 and later this format can only read existing
indexes; use `CuVSCodecs#acceleratedHNSWScalarQuantizedFormat` to write.

## Public Members

### LuceneAcceleratedHNSWScalarQuantizedVectorsFormat

```java
public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat()
```

Initializes `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat` with default values.

**Throws**

| Type | Description |
| --- | --- |
| `LibraryException` | if the native library fails to load |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:29`_

### LuceneAcceleratedHNSWScalarQuantizedVectorsFormat

```java
public LuceneAcceleratedHNSWScalarQuantizedVectorsFormat( AcceleratedHNSWParams acceleratedHNSWParams)
```

Initializes `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat` with the given threads, graph degree, etc.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | An instance of `AcceleratedHNSWParams` |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:38`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsFormat.java:19`_
