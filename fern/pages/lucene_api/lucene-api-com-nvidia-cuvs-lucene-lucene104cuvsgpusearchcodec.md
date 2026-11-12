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

This codec wraps `Lucene104Codec`, the default codec of Lucene 10.4. Once a later Lucene
release replaces that default, this codec can only read existing indexes; use `CuVSCodecs#gpuSearch` to write.

## Public Members

### Lucene104CuVSGPUSearchCodec

```java
public Lucene104CuVSGPUSearchCodec()
```

Creates the codec with default parameters.

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104CuVSGPUSearchCodec.java:23`_

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

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104CuVSGPUSearchCodec.java:33`_

### Lucene104CuVSGPUSearchCodec

```java
public Lucene104CuVSGPUSearchCodec( String name, Codec delegate, GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig)
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
| `params` | GPU index and search parameters |
| `filterCacheConfig` | filter-bitset-cache configuration |

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104CuVSGPUSearchCodec.java:53`_

_Source: `java/cuvs-lucene/lucene-10.4/src/since/java/com/nvidia/cuvs/lucene/Lucene104CuVSGPUSearchCodec.java:19`_
