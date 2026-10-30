---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene103cuvsgpusearchcodec
---

# Lucene103CuVSGPUSearchCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene103CuVSGPUSearchCodec extends CuVSFilterCodec
```

A codec that both builds and searches CAGRA indexes on the GPU. cuVS serialization formats are in
experimental phase and hence backward compatibility cannot be guaranteed.

This codec wraps `Lucene103Codec`, the default codec of Lucene 10.3. Once a later Lucene
release replaces that default, this codec can only read existing indexes; use `CuVSCodecs#gpuSearch` to write.

## Public Members

### Lucene103CuVSGPUSearchCodec

```java
public Lucene103CuVSGPUSearchCodec()
```

Creates the codec with default parameters.

_Source: `java/cuvs-lucene/lucene-10.3/src/since/java/com/nvidia/cuvs/lucene/Lucene103CuVSGPUSearchCodec.java:21`_

### Lucene103CuVSGPUSearchCodec

```java
public Lucene103CuVSGPUSearchCodec( GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig)
```

Creates the codec.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | GPU index and search parameters |
| `filterCacheConfig` | filter-bitset-cache configuration |

_Source: `java/cuvs-lucene/lucene-10.3/src/since/java/com/nvidia/cuvs/lucene/Lucene103CuVSGPUSearchCodec.java:31`_

_Source: `java/cuvs-lucene/lucene-10.3/src/since/java/com/nvidia/cuvs/lucene/Lucene103CuVSGPUSearchCodec.java:17`_
