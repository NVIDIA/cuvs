---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-gpuindex
---

# GPUIndex

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class GPUIndex implements Closeable
```

This class holds references to the cuVS Index (Cagra, Brute force, etc.)

## Public Members

### GPUIndex

```java
public GPUIndex( String segmentName, String fieldName, CagraIndex cagraIndex, int maxDocs, BruteForceIndex bruteforceIndex)
```

Initializes an instance of `GPUIndex`

**Parameters**

| Name | Description |
| --- | --- |
| `segmentName` | the name of the segment |
| `fieldName` | the field name |
| `cagraIndex` | reference to the CagraIndex |
| `maxDocs` | the maximum documents |
| `bruteforceIndex` | reference to the BruteForceIndex |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:35`_

### GPUIndex

```java
public GPUIndex(CagraIndex cagraIndex, BruteForceIndex bruteforceIndex)
```

Initializes an instance of `GPUIndex`

**Parameters**

| Name | Description |
| --- | --- |
| `cagraIndex` | reference to the CagraIndex instance |
| `bruteforceIndex` | reference to the BruteForceIndex instance |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:57`_

### getCagraIndex

```java
public CagraIndex getCagraIndex()
```

Gets the reference to the CAGRA index

**Returns**

an instance of CagraIndex

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:67`_

### getBruteforceIndex

```java
public BruteForceIndex getBruteforceIndex()
```

Gets the reference to the Bruteforce index

**Returns**

an instance of BruteForceIndex

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:77`_

### getFieldName

```java
public String getFieldName()
```

Gets the field name

**Returns**

field name

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:87`_

### getSegmentName

```java
public String getSegmentName()
```

Gets the segment name

**Returns**

segment name

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:96`_

### getMaxDocs

```java
public int getMaxDocs()
```

Gets the max docs

**Returns**

the max docs

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:105`_

### close

```java
@Override public void close() throws IOException
```

Closes this stream and releases any resources associated with it.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:121`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUIndex.java:18`_
