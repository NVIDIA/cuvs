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

Builds an HNSW index from HNSW parameters. The graph is built on the GPU and converted to an
HNSW index that can be searched on the CPU. The graph build algorithm is selected automatically
unless explicit ACE parameters are provided.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | The CuVS resources |
| `hnswParams` | Parameters for the HNSW index |
| `dataset` | The dataset to build the index from |

**Returns**

A new HNSW index ready for search

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during building |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:70`_

### materializeToHnswlib

```java
static void materializeToHnswlib( CuVSResources resources, HnswMaterializeParams materializeParams, String layeredArtifactPath, String outputPath, int dim, HnswIndexParams.CuvsDistanceType metric) throws Throwable
```

Materializes a layered HNSW artifact into a standard hnswlib index file on
disk.

Materializes a `GRAPH_ONLY` artifact (graph topology only,
stored in ACE order) plus a local dataset into a standard hnswlib index file,
without ever holding the full materialized index in host memory. The
resulting file is compatible with the original hnswlib library and can be read
back with `hierarchy == CPU`. The element data type is inferred from the
external dataset. GRAPH_ONLY artifacts are currently produced through the C++
API.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | The CuVS resources |
| `materializeParams` | Materialization parameters (dataset path, host-memory budget, threads) |
| `layeredArtifactPath` | Path to the layered HNSW artifact |
| `outputPath` | Path to the hnswlib index file to write |
| `dim` | The dimension of the vectors in the index |
| `metric` | The distance metric used to build the index |

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during materialization |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:99`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:129`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:138`_

### build

```java
HnswIndex build() throws Throwable
```

Builds and returns an instance of CagraIndex.

**Returns**

an instance of CagraIndex

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:145`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndex.java:17`_
