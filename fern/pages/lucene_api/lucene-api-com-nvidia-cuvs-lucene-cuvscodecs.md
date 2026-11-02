---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvscodecs
---

# CuVSCodecs

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class CuVSCodecs
```

Creates the cuvs-lucene codecs and vectors formats that write indexes on the Lucene release this
artifact is built for.

Each cuvs-lucene codec wraps the default codec of one Lucene release, for example `Lucene101AcceleratedHNSWCodec` wraps Lucene 10.2's `Lucene101Codec`, so the codec to write
with changes along with Lucene. Creating codecs through this class instead of naming them keeps
application code unchanged when it moves to another Lucene release and the matching cuvs-lucene
artifact.

Every method throws `IllegalStateException` if the Lucene release in use is not the one
this cuvs-lucene artifact is built for.

## Public Members

### checkLuceneVersion

```java
public static void checkLuceneVersion()
```

Checks that the Lucene release in use is the one this cuvs-lucene artifact is built for.

A cuvs-lucene artifact on the wrong Lucene release does not break Lucene: Lucene keeps
working, and so do indexes that do not use the cuvs-lucene codecs, while using one of these
codecs fails. Applications that would rather not start at all in that case can call this method
during startup.

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if the artifact is not built for the Lucene release in use; the message names the artifact to use instead |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:39`_

### acceleratedHNSW

```java
public static Codec acceleratedHNSW()
```

Returns the codec that builds HNSW graphs on the GPU and searches them on the CPU, with default
parameters.

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:49`_

### acceleratedHNSW

```java
public static Codec acceleratedHNSW(AcceleratedHNSWParams params)
```

Returns the codec that builds HNSW graphs on the GPU and searches them on the CPU.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the index build parameters |

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:59`_

### acceleratedHNSWFormat

```java
public static KnnVectorsFormat acceleratedHNSWFormat()
```

Returns the vectors format used by `#acceleratedHNSW()`, for use with a per-field codec.

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:69`_

### acceleratedHNSWFormat

```java
public static KnnVectorsFormat acceleratedHNSWFormat(AcceleratedHNSWParams params)
```

Returns the vectors format used by `#acceleratedHNSW(AcceleratedHNSWParams)`, for use
with a per-field codec.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the index build parameters |

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:80`_

### acceleratedHNSWScalarQuantized

```java
public static Codec acceleratedHNSWScalarQuantized()
```

Returns the codec that builds HNSW graphs on the GPU over scalar-quantized vectors, with
default parameters.

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:91`_

### acceleratedHNSWScalarQuantized

```java
public static Codec acceleratedHNSWScalarQuantized(AcceleratedHNSWParams params)
```

Returns the codec that builds HNSW graphs on the GPU over scalar-quantized vectors.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the index build parameters |

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:101`_

### acceleratedHNSWScalarQuantizedFormat

```java
public static KnnVectorsFormat acceleratedHNSWScalarQuantizedFormat()
```

Returns the vectors format used by `#acceleratedHNSWScalarQuantized()`, for use with a
per-field codec.

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:112`_

### acceleratedHNSWScalarQuantizedFormat

```java
public static KnnVectorsFormat acceleratedHNSWScalarQuantizedFormat( AcceleratedHNSWParams params)
```

Returns the vectors format used by `#acceleratedHNSWScalarQuantized(AcceleratedHNSWParams)`, for use with a per-field codec.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the index build parameters |

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:123`_

### acceleratedHNSWBinaryQuantized

```java
public static Codec acceleratedHNSWBinaryQuantized()
```

Returns the codec that builds HNSW graphs on the GPU over binary-quantized vectors, with
default parameters.

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:135`_

### acceleratedHNSWBinaryQuantized

```java
public static Codec acceleratedHNSWBinaryQuantized(AcceleratedHNSWParams params)
```

Returns the codec that builds HNSW graphs on the GPU over binary-quantized vectors.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the index build parameters |

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:145`_

### acceleratedHNSWBinaryQuantizedFormat

```java
public static KnnVectorsFormat acceleratedHNSWBinaryQuantizedFormat()
```

Returns the vectors format used by `#acceleratedHNSWBinaryQuantized()`, for use with a
per-field codec.

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:156`_

### acceleratedHNSWBinaryQuantizedFormat

```java
public static KnnVectorsFormat acceleratedHNSWBinaryQuantizedFormat( AcceleratedHNSWParams params)
```

Returns the vectors format used by `#acceleratedHNSWBinaryQuantized(AcceleratedHNSWParams)`, for use with a per-field codec.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the index build parameters |

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:167`_

### gpuSearch

```java
public static Codec gpuSearch()
```

Returns the codec that both builds and searches CAGRA indexes on the GPU, with default
parameters.

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:179`_

### gpuSearch

```java
public static Codec gpuSearch(GPUSearchParams params)
```

Returns the codec that both builds and searches CAGRA indexes on the GPU, with the default
filter-bitset-cache configuration.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the GPU index and search parameters |

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:190`_

### gpuSearch

```java
public static Codec gpuSearch(GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig)
```

Returns the codec that both builds and searches CAGRA indexes on the GPU.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the GPU index and search parameters |
| `filterCacheConfig` | the filter-bitset-cache configuration |

**Returns**

the codec

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:201`_

### gpuSearchFormat

```java
public static KnnVectorsFormat gpuSearchFormat()
```

Returns the vectors format used by `#gpuSearch()`, for use with a per-field codec.

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:211`_

### gpuSearchFormat

```java
public static KnnVectorsFormat gpuSearchFormat(GPUSearchParams params)
```

Returns the vectors format used by `#gpuSearch(GPUSearchParams)`, for use with a
per-field codec.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the GPU index and search parameters |

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:222`_

### gpuSearchFormat

```java
public static KnnVectorsFormat gpuSearchFormat( GPUSearchParams params, FilterBitsetCacheConfig filterCacheConfig)
```

Returns the vectors format used by `#gpuSearch(GPUSearchParams, FilterBitsetCacheConfig)`, for use with a per-field codec.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | the GPU index and search parameters |
| `filterCacheConfig` | the filter-bitset-cache configuration |

**Returns**

the vectors format

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:234`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVSCodecs.java:25`_
