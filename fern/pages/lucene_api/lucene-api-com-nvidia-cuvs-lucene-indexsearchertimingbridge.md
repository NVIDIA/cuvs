---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-indexsearchertimingbridge
---

# IndexSearcherTimingBridge

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class IndexSearcherTimingBridge implements Function<Map<String, Object>, Map<String, Object>>
```

Measures one `IndexSearcher#search(Query,int)` invocation inside the JVM.

The standard `Function` and `Map` types form a narrow bridge for generated Java
bindings that do not wrap this class directly. Each call still represents one ordinary Lucene
query; this class does not add batching or concurrency.

The request map must contain `searcher` (an `IndexSearcher`), `query` (a
`Query`), and `top_k` (an `Integer`). The response contains `top_docs` (a
`TopDocs`) and `elapsed_nanos` (a `Long`). An `IOException` from Lucene is
exposed as an `UncheckedIOException` because `Function.apply` cannot declare checked
exceptions.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/IndexSearcherTimingBridge.java:29`_
