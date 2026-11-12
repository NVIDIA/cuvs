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

This codec wraps `Lucene104Codec`, the default codec of Lucene 10.4. Once a later Lucene
release replaces that default, this codec can only read existing indexes; use `CuVSCodecs#acceleratedHNSWBinaryQuantized` to write.

## Public Members

### Lucene104AcceleratedHNSWBinaryQuantizedCodec

```java
public Lucene104AcceleratedHNSWBinaryQuantizedCodec()
```

Creates the codec with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWBinaryQuantizedCodec.java:24`_

### Lucene104AcceleratedHNSWBinaryQuantizedCodec

```java
public Lucene104AcceleratedHNSWBinaryQuantizedCodec(AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the codec.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWBinaryQuantizedCodec.java:33`_

### Lucene104AcceleratedHNSWBinaryQuantizedCodec

```java
public Lucene104AcceleratedHNSWBinaryQuantizedCodec( String name, Codec delegate, AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the codec with a delegate chosen by the caller, for applications that wrap a codec of
their own. Unlike the other constructors, this never makes the codec read-only: whether it can
write on a later Lucene release depends on the delegate, and on its vectors format, which a
later Lucene release may stop writing too.

**Parameters**

| Name | Description |
| --- | --- |
| `name` | the codec's name, which Lucene records in every segment the codec writes |
| `delegate` | the codec to filter |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWBinaryQuantizedCodec.java:51`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWBinaryQuantizedCodec.java:20`_
