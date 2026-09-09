---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-lucene99acceleratedhnswvectorsformat
---

# Lucene99AcceleratedHNSWVectorsFormat

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Lucene99AcceleratedHNSWVectorsFormat extends KnnVectorsFormat
```

cuVS based KnnVectorsFormat for indexing on GPU and searching on the CPU.

## Public Members

### Lucene99AcceleratedHNSWVectorsFormat

```java
public Lucene99AcceleratedHNSWVectorsFormat()
```

Initializes `Lucene99AcceleratedHNSWVectorsFormat` with an instance
of `AcceleratedHNSWParams` with default parameter values.

**Throws**

| Type | Description |
| --- | --- |
| `LibraryException` | if the native library fails to load |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsFormat.java:56`_

### Lucene99AcceleratedHNSWVectorsFormat

```java
public Lucene99AcceleratedHNSWVectorsFormat(AcceleratedHNSWParams acceleratedHNSWParams)
```

Initializes `Lucene99AcceleratedHNSWVectorsFormat` with an instance
of `AcceleratedHNSWParams`.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | An instance of `AcceleratedHNSWParams` |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsFormat.java:66`_

### fieldsWriter

```java
@Override public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException
```

Returns a KnnVectorsWriter to write the vectors to the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsFormat.java:74`_

### fieldsReader

```java
@Override public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException
```

Returns a KnnVectorsReader to read the vectors from the index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsFormat.java:101`_

### getMaxDimensions

```java
@Override public int getMaxDimensions(String fieldName)
```

Returns the maximum number of vector dimensions supported by this codec for the given field name.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsFormat.java:114`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Lucene99AcceleratedHNSWVectorsFormat.java:27`_
