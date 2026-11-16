---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvs2510gpuvectorsreader
---

# CuVS2510GPUVectorsReader

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class CuVS2510GPUVectorsReader extends KnnVectorsReader
```

KnnVectorsReader instance associated with cuVS format for reading vectors from an index.

## Public and Protected Members

### CuVS2510GPUVectorsReader

```java
public CuVS2510GPUVectorsReader(SegmentReadState state, FlatVectorsReader flatReader) throws IOException
```

Initializes the `CuVS2510GPUVectorsReader`, checks and loads the index.

**Parameters**

| Name | Description |
| --- | --- |
| `state` | instance of the SegmentReadState |
| `flatReader` | instance of the FlatVectorsReader |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O exception |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:101`_

### close

```java
@Override public void close() throws IOException
```

Closes the resources.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:485`_

### checkIntegrity

```java
@Override public void checkIntegrity() throws IOException
```

Checks consistency of this reader.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:519`_

### getFloatVectorValues

```java
@Override public FloatVectorValues getFloatVectorValues(String field) throws IOException
```

Returns the FloatVectorValues for the given field.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:528`_

### getByteVectorValues

```java
@Override public ByteVectorValues getByteVectorValues(String field)
```

Returns the ByteVectorValues for the given field.

This is not supported.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:538`_

### doSearch

```java
@Override protected void doSearch(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException
```

Returns the k nearest neighbor documents using cuVS's CAGRA or brute force algorithm for this field, to the given vector.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:562`_

### doSearch

```java
@Override protected void doSearch(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException
```

Return the k nearest neighbor documents as determined by comparison of their vector values for this field, to the given vector.

This is not supported.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:739`_

### getCagraIndexForField

```java
public CagraIndex getCagraIndexForField(String field) throws IOException
```

Returns the `CagraIndex` for the given field, or `null` if this segment has none
(e.g., when the field is missing or the segment only holds a brute force index).

If this reader was opened for a merge and has not been searched on this field yet, this
loads the field's index onto the GPU first. The index is owned by this reader, which closes it,
so callers must not close it.

**Parameters**

| Name | Description |
| --- | --- |
| `field` | the vector field name |

**Returns**

the CAGRA index, or `null`

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O exception |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:852`_

### getFieldInfos

```java
public FieldInfos getFieldInfos()
```

Gets the instance of FieldInfos.

**Returns**

the instance of FieldInfos

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:870`_

### getCuvsIndexes

```java
public IntObjectHashMap<GPUIndex> getCuvsIndexes()
```

Gets the map of `GPUIndex` objects.

**Returns**

the map of GPU index objects, or `null` if the reader was opened with a merge `IOContext`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:880`_

### getFieldEntries

```java
public IntObjectHashMap<FieldEntry> getFieldEntries()
```

Gets the map of FieldEntry objects that hold the meta information for the field.

**Returns**

the map of FieldEntry objects

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:889`_

### getOffHeapByteSize

```java
@Override public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo)
```

Reports only the raw vectors. Lucene asks for the memory that searches need to avoid page
faults; searches read the raw vectors from their file, but search the cuVS indexes on the GPU.
The file holding the cuVS indexes is only read to load them onto the GPU, so it is left out,
even though it stays open.

_Source: `java/cuvs-lucene/compat/knn-vectors-reader-106-merge-instance/src/main/java/com/nvidia/cuvs/lucene/CompatKnnVectorsReader.java:48`_

### getMergeInstance

```java
@Override public KnnVectorsReader getMergeInstance() throws IOException
```

Returns a reader that reads the raw vectors through their reader's merge instance.

_Source: `java/cuvs-lucene/compat/knn-vectors-reader-106-merge-instance/src/main/java/com/nvidia/cuvs/lucene/CompatKnnVectorsReader.java:54`_

### finishMerge

```java
@Override public void finishMerge() throws IOException
```

Lets the raw vectors reader release what it held for the merge.

_Source: `java/cuvs-lucene/compat/knn-vectors-reader-106-merge-instance/src/main/java/com/nvidia/cuvs/lucene/CompatKnnVectorsReader.java:61`_

### getFloatVectorValues

```java
@Override public FloatVectorValues getFloatVectorValues(String field, DataAccessHint access) throws IOException
```

Reads the raw vectors, which are the float vectors of the fields, as `access` says.

_Source: `java/cuvs-lucene/compat/knn-vectors-reader-106-merge-instance/src/main/java/com/nvidia/cuvs/lucene/CompatKnnVectorsReader.java:67`_

### getVectorCount

```java
@Override public int getVectorCount(FieldInfo fieldInfo) throws IOException
```

Reads the number of vectors from the metadata of the raw vectors.

_Source: `java/cuvs-lucene/compat/knn-vectors-reader-106-merge-instance/src/main/java/com/nvidia/cuvs/lucene/CompatKnnVectorsReader.java:74`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:62`_
