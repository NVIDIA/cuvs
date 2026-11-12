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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:101`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:114`_

### withRawVectorsReader

```java
KnnVectorsReader withRawVectorsReader(FlatVectorsReader flatReader)
```

Returns a reader for a merge that reads the raw vectors through `flatReader` and shares
everything else with the reader this one was created from, which owns what they share: closing
the returned reader does nothing.

Implements `CompatKnnVectorsReader.withRawVectorsReader`, which only the Lucene 10.6
variant declares, so it is not marked `@Override`; earlier releases never call it.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:184`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:267`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:282`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:345`_

### openCagraIndexForMerge

```java
CagraIndex openCagraIndexForMerge(String field) throws IOException
```

Deserializes the CAGRA index of a single field onto the GPU, bypassing the `GPUIndex`
cache. This is how `CuVS2510GPUVectorsWriter` obtains the inputs for the cuVS merge API:
a reader opened with `Context#MERGE` caches no indexes until it is first searched, and
a cached index belongs to the `com.nvidia.cuvs.CuVSResources` of whichever thread opened
the reader (or first searched it, for a merge-opened reader), while the merge API requires
every input to share one resources instance - the merging thread's.

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:364`_

### lazilyLoadedIndexCount

```java
int lazilyLoadedIndexCount()
```

Returns the number of indexes a merge-opened reader has loaded so far. Package-private so that
tests can check that loading is per field.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:444`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:767`_

### maxHits

```java
int maxHits(String field, CagraSearchParams.SearchAlgo searchAlgo)
```

Returns the most hits a search of the given field in this segment returns, or `Integer#MAX_VALUE` if it has no such limit. A segment with a brute force index has none, and
one with only a CAGRA index returns at most `#MAX_CAGRA_K` hits, or `GPUKnnFloatVectorQuery#MAX_SINGLE_CTA_ITOPK` with SINGLE_CTA. Unlike `#getCagraIndexForField`, this never loads an index onto the GPU.

**Parameters**

| Name | Description |
| --- | --- |
| `field` | the vector field name |
| `searchAlgo` | the CAGRA search algorithm the search runs with |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:822`_

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

### getFilterBitsetCache

```java
FilterBitsetCache getFilterBitsetCache()
```

Returns the filter cache owned by the vectors format that created this reader.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:861`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsReader.java:62`_
