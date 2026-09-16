---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cuvs2510gpuvectorswriter
---

# CuVS2510GPUVectorsWriter

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class CuVS2510GPUVectorsWriter extends KnnVectorsWriter
```

extends upon KnnVectorsWriter and has implementation for critical methods like flush, merge etc.

## Public Members

### CAGRA

```java
CAGRA(true, false), /** Builds a Brute Force index. */ BRUTE_FORCE(false, true), /** Builds both - CAGRA and Brute Force indexes. */ CAGRA_AND_BRUTE_FORCE(true, true)
```

Builds a CAGRA index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:89`_

### BRUTE_FORCE

```java
BRUTE_FORCE(false, true), /** Builds both - CAGRA and Brute Force indexes. */ CAGRA_AND_BRUTE_FORCE(true, true)
```

Builds a Brute Force index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:92`_

### CAGRA_AND_BRUTE_FORCE

```java
CAGRA_AND_BRUTE_FORCE(true, true)
```

Builds both - CAGRA and Brute Force indexes.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:95`_

### CuVS2510GPUVectorsWriter

```java
public CuVS2510GPUVectorsWriter( SegmentWriteState state, GPUSearchParams gpuSearchParams, FlatVectorsWriter flatVectorsWriter) throws IOException
```

Initializes `CuVS2510GPUVectorsWriter`.

**Parameters**

| Name | Description |
| --- | --- |
| `state` | instance of the SegmentWriteState |
| `gpuSearchParams` | An instance of `GPUSearchParams` |
| `flatVectorsWriter` | instance of FlatVectorsWriter |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:121`_

### addField

```java
@Override public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException
```

Add new field for indexing.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:160`_

### flush

```java
@Override public void flush(int maxDoc, DocMap sortMap) throws IOException
```

Creates the CAGRA and/or brute force indexes and writes them to the disk.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:303`_

### mergeOneField

```java
@Override public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException
```

Write field for merging.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:685`_

### ramBytesUsed

```java
@Override public long ramBytesUsed()
```

Returns the memory usage of this object in bytes.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:696`_

### finish

```java
@Override public void finish() throws IOException
```

Called once at the end before close.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:708`_

### close

```java
@Override public void close() throws IOException
```

Close the applicable resources.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:728`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CuVS2510GPUVectorsWriter.java:59`_
