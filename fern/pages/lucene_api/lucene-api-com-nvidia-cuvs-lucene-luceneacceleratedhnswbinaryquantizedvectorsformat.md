---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswbinaryquantizedvectorsformat
---

# LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat extends KnnVectorsFormat
```

cuVS based Binary Quantized KnnVectorsFormat for indexing on GPU and searching on the CPU.

## Public Members

### LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat

```java
public LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat()
```

Initializes `LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat` with default values.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:74`_

### LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat

```java
public LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat( AcceleratedHNSWParams acceleratedHNSWParams)
```

Initializes `LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat` with the given threads, graph degree, etc.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | An instance of `AcceleratedHNSWParams` |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:83`_

### fieldsWriter

```java
@Override public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException
```

Returns a KnnVectorsWriter to write the binary quantized vectors to the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:92`_

### fieldsReader

```java
@Override public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException
```

Returns a KnnVectorsReader to read the binary quantized vectors from the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:122`_

### getMaxDimensions

```java
@Override public int getMaxDimensions(String fieldName)
```

Returns the maximum number of vector dimensions supported by this codec for the given field name.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:136`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat.java:25`_
