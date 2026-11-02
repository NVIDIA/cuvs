---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswscalarquantizedcodec
---

# LuceneAcceleratedHNSWScalarQuantizedCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWScalarQuantizedCodec extends CuVSFilterCodec
```

CuVS based codec for GPU based vector search

This codec, named `Lucene101AcceleratedHNSWScalarQuantizedCodec`, wraps `Lucene101Codec`, the default codec of Lucene 10.2. `CuVSCodecs#acceleratedHNSWScalarQuantized` returns the codec of
this family for the Lucene release in use.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedCodec.java:18`_
