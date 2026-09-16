---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene101acceleratedhnswcodec
---

# Lucene101AcceleratedHNSWCodec

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene101AcceleratedHNSWCodec extends FilterCodec
```

A codec that enables GPU-based accelerated HNSW capability and can be used
to accelerated indexing using GPUs and search using CPUs. Fallbacks to CPU
based indexing when used on a machine without a GPU and/or cuVS.

## Public Members

### Lucene101AcceleratedHNSWCodec

```java
public Lucene101AcceleratedHNSWCodec() throws Exception
```

Default constructor for `Lucene101AcceleratedHNSWCodec`.

**Throws**

| Type | Description |
| --- | --- |
| `Exception` |  |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:31`_

### Lucene101AcceleratedHNSWCodec

```java
public Lucene101AcceleratedHNSWCodec(String name, Codec delegate)
```

Constructor for `Lucene101AcceleratedHNSWCodec`.

**Parameters**

| Name | Description |
| --- | --- |
| `name` | the codec's name |
| `delegate` | the delegate codec to filter |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:41`_

### Lucene101AcceleratedHNSWCodec

```java
public Lucene101AcceleratedHNSWCodec(AcceleratedHNSWParams acceleratedHNSWParams) throws Exception
```

Constructor for `Lucene101AcceleratedHNSWCodec`.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | instance of `AcceleratedHNSWParams` |

**Throws**

| Type | Description |
| --- | --- |
| `Exception` | exception |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:52`_

### knnVectorsFormat

```java
@Override public KnnVectorsFormat knnVectorsFormat()
```

Get the configured `KnnVectorsFormat`.

**Returns**

the instance of the `KnnVectorsFormat`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:87`_

### setKnnFormat

```java
public void setKnnFormat(KnnVectorsFormat format)
```

Set the `KnnVectorsFormat`.

**Parameters**

| Name | Description |
| --- | --- |
| `format` | the `KnnVectorsFormat` to set |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:97`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.java:21`_
