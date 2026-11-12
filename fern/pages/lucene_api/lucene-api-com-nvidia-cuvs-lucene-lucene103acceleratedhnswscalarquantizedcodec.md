---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene103acceleratedhnswscalarquantizedcodec
---

# Lucene103AcceleratedHNSWScalarQuantizedCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene103AcceleratedHNSWScalarQuantizedCodec extends CuVSFilterCodec
```

A codec that builds HNSW graphs on the GPU over scalar-quantized vectors, and writes them in
Lucene's HNSW format so that they are searched on the CPU. See `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat`.

This codec wraps `Lucene103Codec`, the default codec of Lucene 10.3. Once a later Lucene
release replaces that default, this codec can only read existing indexes; use `CuVSCodecs#acceleratedHNSWScalarQuantized` to write.

## Public Members

### Lucene103AcceleratedHNSWScalarQuantizedCodec

```java
public Lucene103AcceleratedHNSWScalarQuantizedCodec()
```

Creates the codec with default parameters.

_Source: `java/cuvs-lucene/lucene-10.3/src/since/java/com/nvidia/cuvs/lucene/Lucene103AcceleratedHNSWScalarQuantizedCodec.java:22`_

### Lucene103AcceleratedHNSWScalarQuantizedCodec

```java
public Lucene103AcceleratedHNSWScalarQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the codec.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.3/src/since/java/com/nvidia/cuvs/lucene/Lucene103AcceleratedHNSWScalarQuantizedCodec.java:31`_

_Source: `java/cuvs-lucene/lucene-10.3/src/since/java/com/nvidia/cuvs/lucene/Lucene103AcceleratedHNSWScalarQuantizedCodec.java:18`_
