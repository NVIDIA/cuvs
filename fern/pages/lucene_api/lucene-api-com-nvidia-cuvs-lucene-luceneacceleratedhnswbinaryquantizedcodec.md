---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-luceneacceleratedhnswbinaryquantizedcodec
---

# LuceneAcceleratedHNSWBinaryQuantizedCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class LuceneAcceleratedHNSWBinaryQuantizedCodec extends CuVSFilterCodec
```

CuVS based codec for GPU based vector search

This codec, named `Lucene101AcceleratedHNSWBinaryQuantizedCodec`, wraps `Lucene101Codec`, the default codec of Lucene 10.2. On later Lucene releases it can only read
existing indexes; use `CuVSCodecs#acceleratedHNSWBinaryQuantized` to write.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedCodec.java:18`_
