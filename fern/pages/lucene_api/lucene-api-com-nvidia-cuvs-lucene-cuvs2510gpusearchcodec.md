---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvs2510gpusearchcodec
---

# CuVS2510GPUSearchCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class CuVS2510GPUSearchCodec extends FilterCodec
```

cuVS based codec for GPU based vector search that enables both - indexing and search on the GPU.
cuVS serialization formats are in experimental phase and hence backward compatibility cannot be guaranteed.

## Public Members

### CuVS2510GPUSearchCodec

```java
public CuVS2510GPUSearchCodec() throws Exception
```

Default constructor for `CuVS2510GPUSearchCodec`.

**Throws**

| Type | Description |
| --- | --- |
| `Exception` |  |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:30`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:45`_

### CuVS2510GPUSearchCodec

```java
public CuVS2510GPUSearchCodec(GPUSearchParams params) throws Exception
```

Initialize the codec with an instance of `GPUSearchParams` having either default
or overridden parameter values.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | An instance of `GPUSearchParams` |

**Throws**

| Type | Description |
| --- | --- |
| `Exception` | Exception raised when initializing the codec |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:56`_

### CuVS2510GPUSearchCodec

```java
public CuVS2510GPUSearchCodec(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig) throws Exception
```

Initialize the codec with GPU search and filter-bitset-cache parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | GPU index and search parameters |
| `filterCacheConfig` | filter-bitset-cache configuration |

**Throws**

| Type | Description |
| --- | --- |
| `Exception` | Exception raised when initializing the codec |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:67`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:80`_

### knnVectorsFormat

```java
@Override public KnnVectorsFormat knnVectorsFormat()
```

Get the configured `KnnVectorsFormat`.

**Returns**

the instance of the `KnnVectorsFormat`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:111`_

### setKnnFormat

```java
public void setKnnFormat(KnnVectorsFormat format)
```

Set the `KnnVectorsFormat`.

**Parameters**

| Name | Description |
| --- | --- |
| `format` | the `KnnVectorsFormat` to set |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:121`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.java:20`_
