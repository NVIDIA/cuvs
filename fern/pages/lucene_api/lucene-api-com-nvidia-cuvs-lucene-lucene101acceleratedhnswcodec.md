---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene101acceleratedhnswcodec
---

# Lucene101AcceleratedHNSWCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene101AcceleratedHNSWCodec extends CuVSFilterCodec
```

A codec that enables GPU-based accelerated HNSW capability and can be used
to accelerated indexing using GPUs and search using CPUs. Fallbacks to CPU
based indexing when used on a machine without a GPU and/or cuVS.

This codec wraps `Lucene101Codec`, the default codec of Lucene 10.2. On later Lucene
releases it can only read existing indexes; use `CuVSCodecs#acceleratedHNSW` to write.

## Public Members

### Lucene101AcceleratedHNSWCodec

```java
public Lucene101AcceleratedHNSWCodec()
```

Default constructor for `Lucene101AcceleratedHNSWCodec`.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:25`_

### Lucene101AcceleratedHNSWCodec

```java
public Lucene101AcceleratedHNSWCodec(String name, Codec delegate)
```

Constructor for `Lucene101AcceleratedHNSWCodec`.

**Parameters**

| Name | Description |
| --- | --- |
| `name` | the codec's name |
| `delegate` | the delegate codec to filter |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:35`_

### Lucene101AcceleratedHNSWCodec

```java
public Lucene101AcceleratedHNSWCodec(AcceleratedHNSWParams acceleratedHNSWParams)
```

Constructor for `Lucene101AcceleratedHNSWCodec`.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | instance of `AcceleratedHNSWParams` |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:48`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:19`_
