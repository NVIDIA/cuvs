---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneprovider
---

# LuceneProvider

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneProvider
```

Dynamically loads Lucene format, reader, and writer classes with a fallback mechanism.

## Public Members

### getLuceneBinaryQuantizedVectorsFormatInstance

```java
public FlatVectorsFormat getLuceneBinaryQuantizedVectorsFormatInstance() throws Exception
```

Returns the Lucene 10.2 flat binary-quantized vectors format.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneProvider.java:326`_

### getluceneBinaryQuantizedVectorsFormatInstance

```java
@Deprecated(since = "26.12", forRemoval = false) public FlatVectorsFormat getluceneBinaryQuantizedVectorsFormatInstance() throws Exception
```

Retains the original public spelling for source and binary compatibility.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneProvider.java:349`_

### getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance

```java
public KnnVectorsFormat getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance( int maxConn, int beamWidth) throws Exception
```

Returns the Lucene 10.2 HNSW binary-quantized vectors format.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneProvider.java:355`_

### getLuceneHnswBinaryQuantizedVectorsFormatInstance

```java
@Deprecated(since = "26.12", forRemoval = false) public FlatVectorsFormat getLuceneHnswBinaryQuantizedVectorsFormatInstance( int maxConn, int beamWidth) throws Exception
```

Retains the original JVM method descriptor for binary compatibility.

The legacy API declared `FlatVectorsFormat` as its return type, but Lucene's HNSW
binary-quantized format extends `KnnVectorsFormat` directly. Use `#getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(int, int)`.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneProvider.java:384`_

### getLuceneHnswScalarQuantizedKnnVectorsFormatInstance

```java
public KnnVectorsFormat getLuceneHnswScalarQuantizedKnnVectorsFormatInstance( int maxConn, int beamWidth) throws Exception
```

Returns Lucene's HNSW scalar-quantized vectors format.

**Parameters**

| Name | Description |
| --- | --- |
| `maxConn` | maximum number of connections per graph node |
| `beamWidth` | number of candidate neighbors tracked while building the graph |

**Returns**

the configured scalar-quantized HNSW format

**Throws**

| Type | Description |
| --- | --- |
| `Exception` | if the Lucene format cannot be constructed |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneProvider.java:418`_

### getLuceneHnswScalarQuantizedVectorsFormatInstance

```java
@Deprecated(since = "26.12", forRemoval = false) public FlatVectorsFormat getLuceneHnswScalarQuantizedVectorsFormatInstance( int beamWidth, int maxConn) throws Exception
```

Retains the original JVM method descriptor for binary compatibility.

The legacy API declared `FlatVectorsFormat` as its return type, but Lucene's HNSW
scalar-quantized format extends `KnnVectorsFormat` directly. Use `#getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(int, int)`. The legacy parameters are
ordered `(beamWidth, maxConn)`; the replacement uses `(maxConn, beamWidth)`.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneProvider.java:448`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneProvider.java:35`_
