---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene104cuvsgpusearchcodec
---

# Lucene104CuVSGPUSearchCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene104CuVSGPUSearchCodec extends CuVSFilterCodec
```

A codec that both builds and searches CAGRA indexes on the GPU. cuVS serialization formats are in
experimental phase and hence backward compatibility cannot be guaranteed.

This codec wraps `Lucene104Codec`, the default codec of Lucene 10.4 and 10.5. Once a
later Lucene release replaces that default, this codec can only read existing indexes; use `CuVSCodecs#gpuSearch` to write.

## Public Members

### Lucene104CuVSGPUSearchCodec

```java
public Lucene104CuVSGPUSearchCodec()
```

Creates the codec with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104CuVSGPUSearchCodec.java:21`_

### Lucene104CuVSGPUSearchCodec

```java
public Lucene104CuVSGPUSearchCodec( GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig)
```

Creates the codec.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | GPU index and search parameters |
| `filterCacheConfig` | filter-bitset-cache configuration |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104CuVSGPUSearchCodec.java:31`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104CuVSGPUSearchCodec.java:17`_
