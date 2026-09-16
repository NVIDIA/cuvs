---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvs2510gpuvectorsreader
---

# CuVS2510GPUVectorsReader

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class CuVS2510GPUVectorsReader extends KnnVectorsReader
```

KnnVectorsReader instance associated with cuVS format for reading vectors from an index.

## Public Members

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:87`_

### CuVS2510GPUVectorsReader

```java
CuVS2510GPUVectorsReader( SegmentReadState state, FlatVectorsReader flatReader, FilterBitsetCache filterBitsetCache) throws IOException
```

Initializes the reader with the cache owned by its vectors format.

**Parameters**

| Name | Description |
| --- | --- |
| `state` | instance of the SegmentReadState |
| `flatReader` | instance of the FlatVectorsReader |
| `filterBitsetCache` | filter cache shared by readers from the same vectors format |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O exception |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:100`_

### readSimilarityFunction

```java
static VectorSimilarityFunction readSimilarityFunction(DataInput input) throws IOException
```

Checks the distance function validity and returns it.

**Parameters**

| Name | Description |
| --- | --- |
| `input` | instance of DataInput |

**Returns**

an instance of VectorSimilarityFunction

**Throws**

| Type | Description |
| --- | --- |
| `IOException` |  |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:228`_

### readVectorEncoding

```java
static VectorEncoding readVectorEncoding(DataInput input) throws IOException
```

Reads the vector encoding (The numeric data type of the vector values) from the DataInput.

**Parameters**

| Name | Description |
| --- | --- |
| `input` | instance of DataInput |

**Returns**

the vector encoding

**Throws**

| Type | Description |
| --- | --- |
| `IOException` |  |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:243`_

### getFieldEntry

```java
FieldEntry getFieldEntry(String field)
```

Gets the FieldEntry for the given field name, or `null` when this segment holds no
cuVS index for it.

**Parameters**

| Name | Description |
| --- | --- |
| `field` | name of the field |

**Returns**

the meta information for the field, or `null`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:306`_

### openCagraIndexForMerge

```java
CagraIndex openCagraIndexForMerge(String field) throws IOException
```

Deserializes the CAGRA index of a single field onto the GPU, bypassing the `GPUIndex`
cache. This is how `CuVS2510GPUVectorsWriter` obtains the inputs for the cuVS merge API:
a reader opened with `Context#MERGE` has no cached indexes at all, and a cached index
belongs to the `com.nvidia.cuvs.CuVSResources` of whichever thread opened the reader,
while the merge API requires every input to share one resources instance - the merging
thread's.

The returned index is owned by the caller and must be closed by it.

**Parameters**

| Name | Description |
| --- | --- |
| `field` | name of the field |

**Returns**

a freshly loaded CAGRA index, or `null` if this segment has none for the field

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O exception |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:325`_

### close

```java
@Override public void close() throws IOException
```

Closes the resources.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:394`_

### checkIntegrity

```java
@Override public void checkIntegrity() throws IOException
```

Checks consistency of this reader.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:412`_

### getFloatVectorValues

```java
@Override public FloatVectorValues getFloatVectorValues(String field) throws IOException
```

Returns the FloatVectorValues for the given field.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:421`_

### getByteVectorValues

```java
@Override public ByteVectorValues getByteVectorValues(String field)
```

Returns the ByteVectorValues for the given field.

This is not supported.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:431`_

### search

```java
@Override public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException
```

Returns the k nearest neighbor documents using cuVS's CAGRA or brute force algorithm for this field, to the given vector.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:455`_

### search

```java
@Override public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException
```

Return the k nearest neighbor documents as determined by comparison of their vector values for this field, to the given vector.

This is not supported.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:598`_

### readEntry

```java
static FieldEntry readEntry( IndexInput input, VectorEncoding vectorEncoding, VectorSimilarityFunction similarityFunction) throws IOException
```

Returns an instance of FieldEntry.

**Parameters**

| Name | Description |
| --- | --- |
| `input` | instance of IndexInput |
| `vectorEncoding` | The numeric data type of the vector values |
| `similarityFunction` | Vector similarity function; used in search to return top K most similar vectors to a target vector |

**Returns**

an instance of FieldEntry

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:626`_

### getCagraIndexForField

```java
public CagraIndex getCagraIndexForField(String field)
```

Returns the `CagraIndex` for the given field, or `null` if unavailable
(e.g., during a merge or when the field is missing).

**Parameters**

| Name | Description |
| --- | --- |
| `field` | the vector field name |

**Returns**

the CAGRA index, or `null`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:678`_

### getFilterBitsetCache

```java
FilterBitsetCache getFilterBitsetCache()
```

Returns the filter cache owned by the vectors format that created this reader.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:688`_

### getFieldInfos

```java
public FieldInfos getFieldInfos()
```

Gets the instance of FieldInfos.

**Returns**

the instance of FieldInfos

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:697`_

### getCuvsIndexes

```java
public IntObjectHashMap<GPUIndex> getCuvsIndexes()
```

Gets the map of `GPUIndex` objects.

**Returns**

the map of GPU index objects

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:706`_

### getFieldEntries

```java
public IntObjectHashMap<FieldEntry> getFieldEntries()
```

Gets the map of FieldEntry objects that hold the meta information for the field.

**Returns**

the map of FieldEntry objects

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:715`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:59`_
