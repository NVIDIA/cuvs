---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene104acceleratedhnswbinaryquantizedcodec
---

# Lucene104AcceleratedHNSWBinaryQuantizedCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene104AcceleratedHNSWBinaryQuantizedCodec extends CuVSFilterCodec
```

A codec that builds HNSW graphs on the GPU over binary-quantized vectors, and writes them in
Lucene's HNSW format so that they are searched on the CPU. See `LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat`.

This codec wraps `Lucene104Codec`, the default codec of Lucene 10.4 and 10.5. Once a
later Lucene release replaces that default, this codec can only read existing indexes; use `CuVSCodecs#acceleratedHNSWBinaryQuantized` to write.

## Public Members

### Lucene104AcceleratedHNSWBinaryQuantizedCodec

```java
public Lucene104AcceleratedHNSWBinaryQuantizedCodec()
```

Creates the codec with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWBinaryQuantizedCodec.java:22`_

### Lucene104AcceleratedHNSWBinaryQuantizedCodec

```java
public Lucene104AcceleratedHNSWBinaryQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the codec.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWBinaryQuantizedCodec.java:31`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWBinaryQuantizedCodec.java:18`_
