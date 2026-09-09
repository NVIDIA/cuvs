---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswscalarquantizedvectorswriter
---

# LuceneAcceleratedHNSWScalarQuantizedVectorsWriter

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWScalarQuantizedVectorsWriter extends KnnVectorsWriter
```

This class extends upon the KnnVectorsWriter to enable the creation of GPU-based accelerated
vector search indexes.

## Public Members

### LuceneAcceleratedHNSWScalarQuantizedVectorsWriter

```java
public LuceneAcceleratedHNSWScalarQuantizedVectorsWriter( SegmentWriteState state, AcceleratedHNSWParams acceleratedHNSWParams, FlatVectorsWriter flatVectorsWriter) throws IOException
```

Initializes `LuceneAcceleratedHNSWScalarQuantizedVectorsWriter`

**Parameters**

| Name | Description |
| --- | --- |
| `state` | instance of the `org.apache.lucene.index.SegmentWriteState` |
| `acceleratedHNSWParams` | An instance of `AcceleratedHNSWParams` |
| `flatVectorsWriter` | instance of the `org.apache.lucene.codecs.hnsw.FlatVectorsWriter` |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | IOException |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:89`_

### addField

```java
@Override public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException
```

Add new field for indexing.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:136`_

### flush

```java
@Override public void flush(int maxDoc, DocMap sortMap) throws IOException
```

Build the indexes and writes it to the disk.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:241`_

### mergeOneField

```java
@Override public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException
```

Write field for merging.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:326`_

### finish

```java
@Override public void finish() throws IOException
```

Called once at the end before close.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:357`_

### close

```java
@Override public void close() throws IOException
```

Closes the resources.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:377`_

### ramBytesUsed

```java
@Override public long ramBytesUsed()
```

Returns the memory usage of this object in bytes.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:386`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:56`_
