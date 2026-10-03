---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene104acceleratedhnswscalarquantizedcodec
---

# Lucene104AcceleratedHNSWScalarQuantizedCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene104AcceleratedHNSWScalarQuantizedCodec extends CuVSFilterCodec
```

A codec that builds HNSW graphs on the GPU over scalar-quantized vectors, and writes them in
Lucene's HNSW format so that they are searched on the CPU. See `Lucene104AcceleratedHNSWScalarQuantizedVectorsFormat`.

This codec wraps `Lucene104Codec`, the default codec of Lucene 10.4 and 10.5. Once a
later Lucene release replaces that default, this codec can only read existing indexes; use `CuVSCodecs#acceleratedHNSWScalarQuantized` to write.

## Public Members

### Lucene104AcceleratedHNSWScalarQuantizedCodec

```java
public Lucene104AcceleratedHNSWScalarQuantizedCodec()
```

Creates the codec with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedCodec.java:22`_

### Lucene104AcceleratedHNSWScalarQuantizedCodec

```java
public Lucene104AcceleratedHNSWScalarQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the codec.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedCodec.java:31`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWScalarQuantizedCodec.java:18`_
