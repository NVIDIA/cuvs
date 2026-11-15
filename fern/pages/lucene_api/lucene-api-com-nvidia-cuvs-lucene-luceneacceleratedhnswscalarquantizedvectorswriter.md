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

## Public and Protected Members

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:80`_

### addField

```java
@Override public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException
```

Add new field for indexing.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:128`_

### flush

```java
@Override public void flush(int maxDoc, DocMap sortMap) throws IOException
```

Build the indexes and writes it to the disk.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:232`_

### finish

```java
@Override public void finish() throws IOException
```

Called once at the end before close.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:349`_

### close

```java
@Override public void close() throws IOException
```

Closes the resources.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:369`_

### ramBytesUsed

```java
@Override public long ramBytesUsed()
```

Returns the memory usage of this object in bytes.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:378`_

### mergeOneField

```java
@Override public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException
```

Merges one field, building its index as it does, and returns `null`: nothing is left
for Lucene to run once every field has been merged. On Lucene 10.2 to 10.4, it returns nothing
(`void`).

_Source: `java/cuvs-lucene/compat/knn-vectors-writer-105-iorunnable/src/main/java/com/nvidia/cuvs/lucene/CompatKnnVectorsWriter.java:29`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedVectorsWriter.java:57`_
