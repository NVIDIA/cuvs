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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:24`_

### nativeLibraryPath

```java
default Path nativeLibraryPath()
```

The directory where to extract and install the native library.
Defaults to \{@systemProperty java.io.tmpdir\}.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:32`_

### newCuVSResources

```java
CuVSResources newCuVSResources(Path tempDirectory) throws Throwable
```

Creates a new CuVSResources.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:37`_

### newCuVSResources

```java
default CuVSResources newCuVSResources( Path tempDirectory, Path memoryTrackingCsvPath, Duration memoryTrackingSampleInterval) throws Throwable
```

Creates a new CuVSResources whose memory allocations are tracked and
written as CSV samples from a background thread.

This method is declared as a `default` method so that adding it
does not break binary compatibility with providers compiled against an
earlier version of this interface; the default implementation throws
`UnsupportedOperationException` and providers must override it to
opt in.

**Parameters**

| Name | Description |
| --- | --- |
| `tempDirectory` | the temporary directory to use for intermediate operations |
| `memoryTrackingCsvPath` | path to the output CSV file (created/truncated) |
| `memoryTrackingSampleInterval` | minimum interval between successive CSV samples |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:56`_

### newHostMatrixBuilder

```java
CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder( long size, long dimensions, CuVSMatrix.DataType dataType)
```

Create a `CuVSMatrix.Builder` instance for a host memory matrix *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:64`_

### newHostMatrixBuilder

```java
CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder( long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType)
```

Create a `CuVSMatrix.Builder` instance for a host memory matrix *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:68`_

### newDeviceMatrixBuilder

```java
CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder( CuVSResources cuVSResources, long size, long dimensions, CuVSMatrix.DataType dataType)
```

Create a `CuVSMatrix.Builder` instance for a device memory matrix *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:72`_

### newDeviceMatrixBuilder

```java
CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder( CuVSResources cuVSResources, long size, long dimensions, int rowStride, int columnStride, CuVSMatrix.DataType dataType)
```

Create a `CuVSMatrix.Builder` instance for a device memory matrix *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:76`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:99`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:118`_

### newMatrixFromArray

```java
CuVSMatrix newMatrixFromArray(float[][] vectors)
```

Create a `CuVSMatrix` from an on-heap array *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:121`_

### newMatrixFromArray

```java
CuVSMatrix newMatrixFromArray(int[][] vectors)
```

Create a `CuVSMatrix` from an on-heap array *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:124`_

### newMatrixFromArray

```java
CuVSMatrix newMatrixFromArray(byte[][] vectors)
```

Create a `CuVSMatrix` from an on-heap array *

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:127`_

### newBruteForceIndexBuilder

```java
BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources) throws UnsupportedOperationException
```

Creates a new BruteForceIndex Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:130`_

### newCagraIndexBuilder

```java
CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources) throws UnsupportedOperationException
```

Creates a new CagraIndex Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:134`_

### newHnswIndexBuilder

```java
HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources) throws UnsupportedOperationException
```

Creates a new HnswIndex Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:138`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:149`_

### hnswIndexBuild

```java
HnswIndex hnswIndexBuild(CuVSResources resources, HnswIndexParams hnswParams, CuVSMatrix dataset) throws Throwable
```

Builds an HNSW index from HNSW parameters using GPU graph construction.

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:160`_

### newTieredIndexBuilder

```java
TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources) throws UnsupportedOperationException
```

Creates a new TieredIndex Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:164`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:174`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:184`_

### isCagraPaddedDataset

```java
default boolean isCagraPaddedDataset(CuVSMatrix dataset)
```

Reports whether the rows of `dataset` already sit at the row stride CAGRA requires, which
is the row length in bytes rounded up to a 16 byte boundary.

This is the question that decides which of the two padded dataset factories a caller has to
use: `CagraIndex#makePaddedDatasetView(CuVSMatrix)` for a device matrix that is already at
that stride, and `CagraIndex#makePaddedDataset(CuVSMatrix)` for one that is not. Asking
for the wrong one is an error rather than an inefficiency, and the stride of a matrix is not
visible outside this library, so callers cannot answer it for themselves.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | the matrix to inspect |

**Returns**

true when the rows are already padded the way CAGRA requires

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | if this provider cannot answer |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:204`_

### newFilterBitsetHandle

```java
FilterBitsetHandle newFilterBitsetHandle(long[] combinedLongs)
```

Creates a device-backed multi-partition filter handle from the pre-packed combined bitset.
Per-partition bit offsets are recomputed inside cuVS from the index sizes.

**Parameters**

| Name | Description |
| --- | --- |
| `combinedLongs` | packed bitset words for a single partition |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:215`_

### searchCagraMultiPartition

```java
MultiPartitionSearchResults searchCagraMultiPartition( CuVSResources resources, List<CagraIndex> indices, CagraQuery query, int k, List<FilterBitsetHandle> filters) throws Throwable
```

Searches multiple CAGRA index partitions for the global top-k nearest neighbors per query.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | shared resources handle |
| `indices` | one CAGRA index per partition, in partition order |
| `query` | query whose vectors are searched against every partition |
| `k` | number of global nearest neighbors to return per query |
| `filters` | one filter per partition (same order as `indices`), or `null`/empty for unfiltered search; a `null` entry means no filter for that partition |

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the search |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:228`_

### gpuInfoProvider

```java
GPUInfoProvider gpuInfoProvider()
```

Returns a `GPUInfoProvider` to query the system for GPU related information

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:237`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:251`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:261`_

### enableRMMAsyncMemory

```java
void enableRMMAsyncMemory()
```

Switch RMM allocations to use stream-ordered asynchronous allocation
(`cudaMallocAsync` / `cudaFreeAsync`). Unlike the pool resource, this resource
returns memory to the stream without blocking the CPU, eliminating device-wide synchronization
on deallocation. This is especially beneficial when multiple CAGRA searches run concurrently
on separate CUDA streams, because internal workspace allocations no longer serialize kernel
launches. This operation has a global effect and will affect all resources on the current device.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:271`_

### resetRMMPooledMemory

```java
void resetRMMPooledMemory()
```

Disables pooled memory on the current device, reverting back to the default setting.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:274`_

### provider

```java
static CuVSProvider provider()
```

Retrieves the system-wide provider.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:277`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:297`_

### cagraIndexParamsFromDataset

```java
CagraIndexParams cagraIndexParamsFromDataset( long rows, long dim, long graphDegree, CagraIndexParams.CuvsDistanceType metric, long buildQuality)
```

Create CAGRA index parameters heuristically tuned for a dataset.

**Parameters**

| Name | Description |
| --- | --- |
| `rows` | The number of rows in the input dataset |
| `dim` | The number of dimensions in the input dataset |
| `graphDegree` | Degree of the output graph |
| `metric` | The distance metric to search |
| `buildQuality` | Higher values increase build quality (and cost) up to a point |

**Returns**

A new CAGRA index parameters object

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:315`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSProvider.java:17`_
