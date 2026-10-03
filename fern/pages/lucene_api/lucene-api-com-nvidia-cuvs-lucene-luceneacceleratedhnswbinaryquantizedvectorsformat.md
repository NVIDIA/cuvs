---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswbinaryquantizedvectorsformat
---

# LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat extends KnnVectorsFormat
```

cuVS based Binary Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.

The vectors are binary quantized only to build the graph on the GPU. The index stores them in
full precision, with Lucene's `Lucene99FlatVectorsFormat`, and search scores them in full
precision too.

## Public Members

### LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat

```java
public LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat()
```

Initializes `LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat` with default values.

**Throws**

| Type | Description |
| --- | --- |
| `LibraryException` | if the native library fails to load |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:46`_

### LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat

```java
public LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat( AcceleratedHNSWParams acceleratedHNSWParams)
```

Initializes `LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat` with the given threads, graph degree, etc.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | An instance of `AcceleratedHNSWParams` |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:55`_

### fieldsWriter

```java
@Override public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException
```

Returns a KnnVectorsWriter to write the binary quantized vectors to the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:64`_

### fieldsReader

```java
@Override public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException
```

Returns a KnnVectorsReader to read the binary quantized vectors from the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:86`_

### getMaxDimensions

```java
@Override public int getMaxDimensions(String fieldName)
```

Returns the maximum number of vector dimensions supported by this codec for the given field name.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:95`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:32`_
