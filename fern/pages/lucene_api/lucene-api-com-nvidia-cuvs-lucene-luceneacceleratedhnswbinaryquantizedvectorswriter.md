---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswbinaryquantizedvectorswriter
---

# LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter extends KnnVectorsWriter
```

This class extends upon the KnnVectorsWriter to enable the creation of GPU-based accelerated
vector search indexes.

## Public Members

### LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter

```java
public LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter( SegmentWriteState state, AcceleratedHNSWParams acceleratedHNSWParams, FlatVectorsWriter flatVectorsWriter) throws IOException
```

Initializes `LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter`

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter.java:80`_

### addField

```java
@Override public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException
```

Add new field for indexing.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter.java:129`_

### flush

```java
@Override public void flush(int maxDoc, DocMap sortMap) throws IOException
```

Build the indexes and writes it to the disk.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter.java:233`_

### finish

```java
@Override public void finish() throws IOException
```

Called once at the end before close.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter.java:348`_

### close

```java
@Override public void close() throws IOException
```

Closes the resources.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter.java:368`_

### ramBytesUsed

```java
@Override public long ramBytesUsed()
```

Returns the memory usage of this object in bytes.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter.java:377`_

### mergeOneField

```java
@Override public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException
```

Merges one field, building its index as it does, and returns `null`: nothing is left
for Lucene to run once every field has been merged.

_Source: `java/cuvs-lucene/compat/knn-vectors-writer-105-iorunnable/src/main/java/com/nvidia/cuvs/lucene/CompatKnnVectorsWriter.java:28`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter.java:57`_
