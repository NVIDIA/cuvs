---
slug: api-reference/java-api-com-nvidia-cuvs-hnswindex
---

# HnswIndex

_Java package: `com.nvidia.cuvs`_

```java
public interface HnswIndex extends AutoCloseable
```

`HnswIndex` encapsulates a HNSW index, along with methods to interact
with it.

## Public Members

### close

```java
@Override void close() throws Exception
```

Invokes the native destroy_hnsw_index to de-allocate the HNSW index

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:21`_

### search

```java
SearchResults search(HnswQuery query) throws Throwable
```

Invokes the native search_hnsw_index via the Panama API for searching a HNSW
index.

**Parameters**

| Name | Description |
| --- | --- |
| `query` | an instance of `HnswQuery` holding the query vectors and other parameters |

**Returns**

an instance of `SearchResults` containing the results

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:32`_

### newBuilder

```java
static HnswIndex.Builder newBuilder(CuVSResources cuvsResources)
```

Creates a new Builder with an instance of `CuVSResources`.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsResources` | an instance of `CuVSResources` |

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | if the provider does not cuvs |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:40`_

### fromCagra

```java
static HnswIndex fromCagra(HnswIndexParams hnswParams, CagraIndex cagraIndex) throws Throwable
```

Creates an HNSW index from an existing CAGRA index.

**Parameters**

| Name | Description |
| --- | --- |
| `hnswParams` | Parameters for the HNSW index |
| `cagraIndex` | The CAGRA index to convert from |

**Returns**

A new HNSW index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during conversion |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:53`_

### build

```java
static HnswIndex build(CuVSResources resources, HnswIndexParams hnswParams, CuVSMatrix dataset) throws Throwable
```

Builds an HNSW index using the ACE (Augmented Core Extraction) algorithm.

ACE enables building HNSW indexes for datasets too large to fit in GPU
memory by partitioning the dataset and building sub-indexes for each
partition independently.

NOTE: This method requires `hnswParams.getAceParams()` to be set with
an instance of HnswAceParams.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | The CuVS resources |
| `hnswParams` | Parameters for the HNSW index with ACE configuration |
| `dataset` | The dataset to build the index from |

**Returns**

A new HNSW index ready for search

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during building |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:75`_

### from

```java
Builder from(InputStream inputStream)
```

Sets an instance of InputStream typically used when index deserialization is
needed.

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | an instance of `InputStream` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:96`_

### withIndexParams

```java
Builder withIndexParams(HnswIndexParams hnswIndexParameters)
```

Registers an instance of configured `HnswIndexParams` with this
Builder.

**Parameters**

| Name | Description |
| --- | --- |
| `hnswIndexParameters` | An instance of HnswIndexParams. |

**Returns**

An instance of this Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:105`_

### build

```java
HnswIndex build() throws Throwable
```

Builds and returns an instance of CagraIndex.

**Returns**

an instance of CagraIndex

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:112`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:17`_
