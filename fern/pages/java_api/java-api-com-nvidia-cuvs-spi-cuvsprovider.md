---
slug: api-reference/java-api-com-nvidia-cuvs-spi-cuvsprovider
---

# CuVSProvider

_Java package: `com.nvidia.cuvs.spi`_

```java
public interface CuVSProvider
```

A provider of low-level cuvs resources and builders.

## Public Members

### tempDirectory

```java
static Path tempDirectory()
```

The temporary directory to use for intermediate operations.
Defaults to \{@systemProperty java.io.tmpdir\}.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:22`_

### nativeLibraryPath

```java
default Path nativeLibraryPath()
```

The directory where to extract and install the native library.
Defaults to \{@systemProperty java.io.tmpdir\}.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:30`_

### newCuVSResources

```java
CuVSResources newCuVSResources(Path tempDirectory) throws Throwable
```

Creates a new CuVSResources.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:35`_

### newHostMatrixBuilder

```java
CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder( long size, long dimensions, CuVSMatrix.DataType dataType)
```

Create a `CuVSMatrix.Builder` instance for a host memory matrix *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:38`_

### newHostMatrixBuilder

```java
CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder( long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType)
```

Create a `CuVSMatrix.Builder` instance for a host memory matrix *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:42`_

### newDeviceMatrixBuilder

```java
CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder( CuVSResources cuVSResources, long size, long dimensions, CuVSMatrix.DataType dataType)
```

Create a `CuVSMatrix.Builder` instance for a device memory matrix *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:46`_

### newDeviceMatrixBuilder

```java
CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder( CuVSResources cuVSResources, long size, long dimensions, int rowStride, int columnStride, CuVSMatrix.DataType dataType)
```

Create a `CuVSMatrix.Builder` instance for a device memory matrix *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:50`_

### newNativeMatrixBuilder

```java
MethodHandle newNativeMatrixBuilder()
```

Returns the factory method used to build a CuVSMatrix from native memory.
The factory method will have this signature:
`CuVSMatrix createNativeMatrix(memorySegment, size, dimensions, dataType)`,
where `memorySegment` is a `java.lang.foreign.MemorySegment` containing `int size` vectors of
`int dimensions` length of type `CuVSMatrix.DataType`.

In order to expose this factory in a way that is compatible with Java 21, the factory method is returned as a
`MethodHandle` with `MethodType` equal to
`(CuVSMatrix.class, MemorySegment.class, int.class, int.class, CuVSMatrix.DataType.class)`.
The caller will need to invoke the factory via the `MethodHandle#invokeExact` method:
`var matrix = (CuVSMatrix)newNativeMatrixBuilder().invokeExact(memorySegment, size, dimensions, dataType)`

**Returns**

a MethodHandle which can be invoked to build a CuVSMatrix from an external `MemorySegment`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:73`_

### newNativeMatrixBuilderWithStrides

```java
MethodHandle newNativeMatrixBuilderWithStrides()
```

Returns the factory method used to build a CuVSMatrix from native memory, with strides.
The factory method will have this signature:
`CuVSMatrix createNativeMatrix(memorySegment, size, dimensions, rowStride, columnStride, dataType)`,
where `memorySegment` is a `java.lang.foreign.MemorySegment` containing `int size` vectors of
`int dimensions` length of type `CuVSMatrix.DataType`. Rows have a stride of `rowStride`,
where 0 indicates "no stride" (a stride equal to the number of columns), and columns have a stride of
`columnStride`

In order to expose this factory in a way that is compatible with Java 21, the factory method is returned as a
`MethodHandle` with `MethodType` equal to
`(CuVSMatrix.class, MemorySegment.class, int.class, int.class, int.class, int.class, DataType.class)`.
The caller will need to invoke the factory via the `MethodHandle#invokeExact` method:
`var matrix = (CuVSMatrix)newNativeMatrixBuilder().invokeExact(memorySegment, size, dimensions, rowStride, columnStride, dataType)`

**Returns**

a MethodHandle which can be invoked to build a CuVSMatrix from an external `MemorySegment`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:92`_

### newMatrixFromArray

```java
CuVSMatrix newMatrixFromArray(float[][] vectors)
```

Create a `CuVSMatrix` from an on-heap array *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:95`_

### newMatrixFromArray

```java
CuVSMatrix newMatrixFromArray(int[][] vectors)
```

Create a `CuVSMatrix` from an on-heap array *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:98`_

### newMatrixFromArray

```java
CuVSMatrix newMatrixFromArray(byte[][] vectors)
```

Create a `CuVSMatrix` from an on-heap array *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:101`_

### newBruteForceIndexBuilder

```java
BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources) throws UnsupportedOperationException
```

Creates a new BruteForceIndex Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:104`_

### newCagraIndexBuilder

```java
CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources) throws UnsupportedOperationException
```

Creates a new CagraIndex Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:108`_

### newHnswIndexBuilder

```java
HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources) throws UnsupportedOperationException
```

Creates a new HnswIndex Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:112`_

### hnswIndexFromCagra

```java
HnswIndex hnswIndexFromCagra(HnswIndexParams hnswParams, CagraIndex cagraIndex) throws Throwable
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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:123`_

### hnswIndexBuild

```java
HnswIndex hnswIndexBuild(CuVSResources resources, HnswIndexParams hnswParams, CuVSMatrix dataset) throws Throwable
```

Builds an HNSW index using the ACE (Augmented Core Extraction) algorithm.

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:134`_

### newTieredIndexBuilder

```java
TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources) throws UnsupportedOperationException
```

Creates a new TieredIndex Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:138`_

### mergeCagraIndexes

```java
CagraIndex mergeCagraIndexes(CagraIndex[] indexes) throws Throwable
```

Merges multiple CAGRA indexes into a single index.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:148`_

### mergeCagraIndexes

```java
default CagraIndex mergeCagraIndexes(CagraIndex[] indexes, CagraIndexParams mergeParams) throws Throwable
```

Merges multiple CAGRA indexes into a single index with the specified merge parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:158`_

### gpuInfoProvider

```java
GPUInfoProvider gpuInfoProvider()
```

Returns a `GPUInfoProvider` to query the system for GPU related information

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:165`_

### enableRMMPooledMemory

```java
void enableRMMPooledMemory(int initialPoolSizePercent, int maxPoolSizePercent)
```

Switch RMM allocations (used internally by various cuVS algorithms and by the default implementation of
`CuVSDeviceMatrix`) to use pooled memory.
This operation has a global effect, and will affect all resources on the current device.

**Parameters**

| Name | Description |
| --- | --- |
| `initialPoolSizePercent` | The initial pool size, in percentage of the total GPU memory |
| `maxPoolSizePercent` | The maximum pool size, in percentage of the total GPU memory |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:179`_

### enableRMMManagedPooledMemory

```java
void enableRMMManagedPooledMemory(int initialPoolSizePercent, int maxPoolSizePercent)
```

Switch RMM allocations (used internally by various cuVS algorithms and by the default implementation of
`CuVSDeviceMatrix`) to use pooled memory.
This operation has a global effect, and will affect all resources on the current device.

**Parameters**

| Name | Description |
| --- | --- |
| `initialPoolSizePercent` | The initial pool size, in percentage of the total GPU memory |
| `maxPoolSizePercent` | The maximum pool size, in percentage of the total GPU memory |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:189`_

### resetRMMPooledMemory

```java
void resetRMMPooledMemory()
```

Disables pooled memory on the current device, reverting back to the default setting.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:192`_

### provider

```java
static CuVSProvider provider()
```

Retrieves the system-wide provider.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:195`_

### cagraIndexParamsFromHnswParams

```java
CagraIndexParams cagraIndexParamsFromHnswParams( long rows, long dim, int m, int efConstruction, CagraIndexParams.HnswHeuristicType heuristic, CagraIndexParams.CuvsDistanceType metric)
```

Create a CAGRA index parameters compatible with HNSW index

Note: The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce
exactly the same recalls and QPS for the same parameter `ef`. The graphs are different
internally. Depending on the selected heuristics, the CAGRA-produced graph's QPS-Recall curve
may be shifted along the curve right or left. See the heuristics descriptions for more details.

**Parameters**

| Name | Description |
| --- | --- |
| `rows` | The number of rows in the input dataset |
| `dim` | The number of dimensions in the input dataset |
| `m` | HNSW index parameter M |
| `efConstruction` | HNSW index parameter ef_construction |
| `heuristic` | The heuristic to use for selecting the graph build parameters |
| `metric` | The distance metric to search |

**Returns**

A new CAGRA index parameters object

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:215`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:15`_
