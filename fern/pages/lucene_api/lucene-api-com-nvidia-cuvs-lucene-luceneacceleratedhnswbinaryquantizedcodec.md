---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswbinaryquantizedcodec
---

# LuceneAcceleratedHNSWBinaryQuantizedCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWBinaryQuantizedCodec extends CuVSFilterCodec
```

CuVS based codec for GPU based vector search

This codec, named `Lucene101AcceleratedHNSWBinaryQuantizedCodec`, wraps `Lucene101Codec`, the default codec of Lucene 10.2. `CuVSCodecs#acceleratedHNSWBinaryQuantized` returns the codec of
this family for the Lucene release in use.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedCodec.java:18`_
