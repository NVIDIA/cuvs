---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene99acceleratedhnswvectorswriter
---

# Lucene99AcceleratedHNSWVectorsWriter

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene99AcceleratedHNSWVectorsWriter extends KnnVectorsWriter
```

This class extends upon the KnnVectorsWriter to
enable the creation of GPU-based accelerated HNSW based vector search.

## Public Members

### Lucene99AcceleratedHNSWVectorsWriter

```java
public Lucene99AcceleratedHNSWVectorsWriter( SegmentWriteState state, AcceleratedHNSWParams acceleratedHNSWParams, FlatVectorsWriter flatVectorsWriter) throws IOException
```

Initializes `Lucene99AcceleratedHNSWVectorsWriter`

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsWriter.java:89`_

### addField

```java
@Override public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException
```

Add new field for indexing.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsWriter.java:131`_

### flush

```java
@Override public void flush(int maxDoc, DocMap sortMap) throws IOException
```

Build the indexes and writes it to the disk.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsWriter.java:226`_

### mergeOneField

```java
@Override public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException
```

Write field for merging.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsWriter.java:339`_

### finish

```java
@Override public void finish() throws IOException
```

Called once at the end before close.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsWriter.java:348`_

### close

```java
@Override public void close() throws IOException
```

Closes the resources.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsWriter.java:368`_

### ramBytesUsed

```java
@Override public long ramBytesUsed()
```

Returns the memory usage of this object in bytes.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsWriter.java:378`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsWriter.java:55`_
