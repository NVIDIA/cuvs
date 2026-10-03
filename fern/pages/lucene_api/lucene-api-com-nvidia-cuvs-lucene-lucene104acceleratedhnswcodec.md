---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene104acceleratedhnswcodec
---

# Lucene104AcceleratedHNSWCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene104AcceleratedHNSWCodec extends CuVSFilterCodec
```

A codec that builds HNSW graphs on the GPU, and writes them in Lucene's HNSW format so that they
are searched on the CPU. Falls back to building the graph on the CPU when no GPU or cuVS is
available.

This codec wraps `Lucene104Codec`, the default codec of Lucene 10.4 and 10.5. Once a
later Lucene release replaces that default, this codec can only read existing indexes; use `CuVSCodecs#acceleratedHNSW` to write.

## Public Members

### Lucene104AcceleratedHNSWCodec

```java
public Lucene104AcceleratedHNSWCodec()
```

Creates the codec with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWCodec.java:22`_

### Lucene104AcceleratedHNSWCodec

```java
public Lucene104AcceleratedHNSWCodec(AcceleratedHNSWParams acceleratedHNSWParams)
```

Creates the codec.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the index build parameters |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWCodec.java:31`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104AcceleratedHNSWCodec.java:18`_
