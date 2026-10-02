---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene104acceleratedhnswscalarquantizedvectorsformat
---

# Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat extends BaseAcceleratedHNSWScalarQuantizedVectorsFormat
```

cuVS based scalar-quantized KnnVectorsFormat for indexing on the GPU and searching on the CPU.

Stores the vectors with Lucene's `Lucene104ScalarQuantizedVectorsFormat`, quantized to 7
bits per dimension, which replaced `Lucene99ScalarQuantizedVectorsFormat` in Lucene 10.4.
The graph is built on the GPU over the vectors quantized the same way as `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat` does.

## Public Members

### Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

```java
public Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat()
```

Creates the format with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:23`_

### Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat

```java
public Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat( AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the format.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:32`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat.java:17`_
