---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene101configuredhnswcodec
---

# Lucene101ConfiguredHNSWCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public final class Lucene101ConfiguredHNSWCodec extends Lucene101AcceleratedHNSWCodec
```

A no-argument `Lucene101AcceleratedHNSWCodec` configured through JVM system properties.

Maximum-connections property: `com.nvidia.cuvs.lucene.hnsw.maxConn`

Beam-width property: `com.nvidia.cuvs.lucene.hnsw.beamWidth`

Set both before construction. Each constructor call snapshots both values, so later property
changes affect only later instances. Because JVM system properties are process-global, callers
that change them from multiple threads must serialize setting both properties and constructing
the codec.

This class is intentionally not a Lucene SPI provider. Load it explicitly by class name when
a binding can invoke only a public no-argument constructor.

## Public Members

### Lucene101ConfiguredHNSWCodec

```java
public Lucene101ConfiguredHNSWCodec() throws Exception
```

Constructs an accelerated HNSW codec from the required JVM system properties.

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if either required property is absent |
| `IllegalArgumentException` | if either property is not an integer in its supported range |
| `Exception` | if the delegated codec cannot be constructed |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101ConfiguredHNSWCodec.java:42`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101ConfiguredHNSWCodec.java:26`_
