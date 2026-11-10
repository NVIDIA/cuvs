---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswscalarquantizedcodec
---

# LuceneAcceleratedHNSWScalarQuantizedCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWScalarQuantizedCodec extends CuVSFilterCodec
```

CuVS based codec for GPU based vector search

This codec, named `Lucene101AcceleratedHNSWScalarQuantizedCodec`, wraps `Lucene101Codec`, the default codec of Lucene 10.2. On later Lucene releases it can only read
existing indexes; use `CuVSCodecs#acceleratedHNSWScalarQuantized` to write.

## Public Members

### LuceneAcceleratedHNSWScalarQuantizedCodec

```java
public LuceneAcceleratedHNSWScalarQuantizedCodec(String name, Codec delegate)
```

Creates the codec with a delegate chosen by the caller, for applications that wrap a codec of
their own. Its vectors format, `LuceneAcceleratedHNSWScalarQuantizedVectorsFormat`, can
only read from Lucene 10.4 on, so this codec can then only read too, whatever the delegate;
write with the codec `CuVSCodecs#acceleratedHNSWScalarQuantized` returns instead, whose
class also takes a delegate.

**Parameters**

| Name | Description |
| --- | --- |
| `name` | the codec's name, which Lucene records in every segment the codec writes |
| `delegate` | the codec to filter |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedCodec.java:35`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedCodec.java:18`_
