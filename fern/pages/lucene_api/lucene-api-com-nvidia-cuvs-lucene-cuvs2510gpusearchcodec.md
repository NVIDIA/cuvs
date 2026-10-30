---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvs2510gpusearchcodec
---

# CuVS2510GPUSearchCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class CuVS2510GPUSearchCodec extends CuVSFilterCodec
```

cuVS based codec for GPU based vector search that enables both - indexing and search on the GPU.
cuVS serialization formats are in experimental phase and hence backward compatibility cannot be guaranteed.

This codec wraps `Lucene101Codec`, the default codec of Lucene 10.2. On later Lucene
releases it can only read existing indexes; use `CuVSCodecs#gpuSearch` to write.

## Public Members

### CuVS2510GPUSearchCodec

```java
public CuVS2510GPUSearchCodec()
```

Default constructor for `CuVS2510GPUSearchCodec`.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:24`_

### CuVS2510GPUSearchCodec

```java
public CuVS2510GPUSearchCodec(String name, Codec delegate)
```

Initialize `CuVS2510GPUSearchCodec` with an instance of `GPUSearchParams`
having default parameter values.

**Parameters**

| Name | Description |
| --- | --- |
| `name` | the name of the codec |
| `delegate` | the delegate codec |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:35`_

### CuVS2510GPUSearchCodec

```java
public CuVS2510GPUSearchCodec(GPUSearchParams params)
```

Initialize the codec with an instance of `GPUSearchParams` having either default
or overridden parameter values.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | An instance of `GPUSearchParams` |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:45`_

### CuVS2510GPUSearchCodec

```java
public CuVS2510GPUSearchCodec(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig)
```

Initialize the codec with GPU search and filter-bitset-cache parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | GPU index and search parameters |
| `filterCacheConfig` | filter-bitset-cache configuration |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:55`_

### CuVS2510GPUSearchCodec

```java
public CuVS2510GPUSearchCodec( String name, Codec delegate, GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig)
```

Initialize a named codec with explicit delegate, GPU search, and filter-cache parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `name` | the name of the codec |
| `delegate` | the delegate codec |
| `params` | GPU index and search parameters |
| `filterCacheConfig` | filter-bitset-cache configuration |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:71`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:18`_
