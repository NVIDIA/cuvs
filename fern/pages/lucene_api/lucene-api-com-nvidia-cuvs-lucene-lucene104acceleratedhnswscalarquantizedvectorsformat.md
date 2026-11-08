---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene104acceleratedhnswscalarquantizedvectorsformat
---

# Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat extends KnnVectorsFormat
```

cuVS based scalar-quantized KnnVectorsFormat for indexing on the GPU and searching on the CPU.

Stores the vectors with Lucene's `Lucene104ScalarQuantizedVectorsFormat`, quantized to
one byte per dimension (`UNSIGNED_BYTE`), which replaced `Lucene99ScalarQuantizedVectorsFormat` in Lucene 10.4. On the GPU, the graph is built over the
vectors quantized to 7 bits, as `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat` does,
and the stored encoding only affects how the graph is searched on the CPU. Without a GPU, Lucene
builds the graph on the CPU, scoring with the stored encoding.

## Public Members

### Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

```java
public Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat()
```

Creates the format with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:27`_

### Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

```java
public Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat( AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the format.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:36`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:21`_
