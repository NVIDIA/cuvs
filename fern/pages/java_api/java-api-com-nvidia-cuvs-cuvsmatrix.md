---
slug: api-reference/java-api-com-nvidia-cuvs-cuvsmatrix
---

# CuVSMatrix

_Java package: `com.nvidia.cuvs`_

```java
public interface CuVSMatrix extends AutoCloseable
```

This represents a wrapper for a dataset to be used for index construction.
The purpose is to allow a caller to place the vectors into native memory
directly, instead of requiring the caller to load all the vectors into the heap
(e.g. with a float[][]).

## Public Members

### ofArray

```java
static CuVSMatrix ofArray(float[][] vectors)
```

Creates a dataset from an on-heap array of vectors.
This method will allocate an additional MemorySegment to hold the graph data.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:46`_

### ofArray

```java
static CuVSMatrix ofArray(int[][] vectors)
```

Creates a dataset from an on-heap array of vectors.
This method will allocate an additional MemorySegment to hold the graph data.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:56`_

### ofArray

```java
static CuVSMatrix ofArray(byte[][] vectors)
```

Creates a dataset from an on-heap array of vectors.
This method will allocate an additional MemorySegment to hold the graph data.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:66`_

### addVector

```java
void addVector(float[] vector)
```

Adds a single vector to the matrix.

**Parameters**

| Name | Description |
| --- | --- |
| `vector` | A float array of as many elements as the dimensions |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:80`_

### addVector

```java
void addVector(byte[] vector)
```

Adds a single vector to the matrix.

**Parameters**

| Name | Description |
| --- | --- |
| `vector` | A byte array of as many elements as the dimensions |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:87`_

### addVector

```java
void addVector(int[] vector)
```

Adds a single vector to the matrix.

**Parameters**

| Name | Description |
| --- | --- |
| `vector` | An int array of as many elements as the dimensions |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:94`_

### hostBuilder

```java
static Builder<CuVSHostMatrix> hostBuilder(long size, long columns, DataType dataType)
```

Returns a builder to create a new instance of a host-memory matrix

**Parameters**

| Name | Description |
| --- | --- |
| `size` | Number of rows (e.g. vectors in a dataset) |
| `columns` | Number of columns (e.g. dimension of each vector in the dataset) |
| `dataType` | The data type of the dataset elements |

**Returns**

a builder for creating a `CuVSHostMatrix`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:107`_

### hostBuilder

```java
static Builder<CuVSHostMatrix> hostBuilder( long size, long columns, int rowStride, int columnStride, DataType dataType)
```

Returns a builder to create a new instance of a host-memory matrix

**Parameters**

| Name | Description |
| --- | --- |
| `size` | Number of rows (e.g. vectors in a dataset) |
| `columns` | Number of columns (e.g. dimension of each vector in the dataset) |
| `rowStride` | The stride (in number of elements) for each row. Must be -1 or &gt; than `columns` |
| `columnStride` | The stride for each column. Currently, it is not supported (must be -1) |
| `dataType` | The data type of the dataset elements |

**Returns**

a builder for creating a `CuVSDeviceMatrix`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:121`_

### deviceBuilder

```java
static Builder<CuVSDeviceMatrix> deviceBuilder( CuVSResources resources, long size, long columns, DataType dataType)
```

Returns a builder to create a new instance of a dataset

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | CuVS resources used to allocate the device memory needed |
| `size` | Number of rows (e.g. vectors in a dataset) |
| `columns` | Number of columns (e.g. dimension of each vector in the dataset) |
| `dataType` | The data type of the dataset elements |

**Returns**

a builder for creating a `CuVSDeviceMatrix`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:136`_

### deviceBuilder

```java
static Builder<CuVSDeviceMatrix> deviceBuilder( CuVSResources resources, long size, long columns, int rowStride, int columnStride, DataType dataType)
```

Returns a builder to create a new instance of a dataset

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | CuVS resources used to allocate the device memory needed |
| `size` | Number of rows (e.g. vectors in a dataset) |
| `columns` | Number of columns (e.g. dimension of each vector in the dataset) |
| `rowStride` | The stride (in number of elements) for each row. Must be -1 or &gt; than `columns` |
| `columnStride` | The stride for each column. Currently, it is not supported (must be -1) |
| `dataType` | The data type of the dataset elements |

**Returns**

a builder for creating a `CuVSDeviceMatrix`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:152`_

### size

```java
long size()
```

Gets the size of the dataset

**Returns**

Size of the dataset

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:168`_

### columns

```java
long columns()
```

Gets the number of columns in the Dataset (e.g. the dimensions of the vectors in this dataset,
or the graph degree for the graph represented as a list of neighbours

**Returns**

Dimensions of the vectors in the dataset

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:176`_

### dataType

```java
DataType dataType()
```

Gets the element type

**Returns**

a `DataType` describing the matrix element type

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:183`_

### getRow

```java
RowView getRow(long row)
```

Get a view (0-copy) of the row data, as a list of integers (32 bit)

**Parameters**

| Name | Description |
| --- | --- |
| `row` | the row for which to return the data |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:190`_

### toArray

```java
void toArray(int[][] array)
```

Copies the content of this dataset to an on-heap Java matrix (array of arrays).

**Parameters**

| Name | Description |
| --- | --- |
| `array` | the destination array. Must be of length `CuVSMatrix#size()` or bigger, and each element must be of length `CuVSMatrix#columns()` or bigger. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:198`_

### toArray

```java
void toArray(float[][] array)
```

Copies the content of this dataset to an on-heap Java matrix (array of arrays).

**Parameters**

| Name | Description |
| --- | --- |
| `array` | the destination array. Must be of length `CuVSMatrix#size()` or bigger, and each element must be of length `CuVSMatrix#columns()` or bigger. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:206`_

### toArray

```java
void toArray(byte[][] array)
```

Copies the content of this dataset to an on-heap Java matrix (array of arrays).

**Parameters**

| Name | Description |
| --- | --- |
| `array` | the destination array. Must be of length `CuVSMatrix#size()` or bigger, and each element must be of length `CuVSMatrix#columns()` or bigger. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:214`_

### toHost

```java
void toHost(CuVSHostMatrix hostMatrix)
```

Fills the provided, pre-allocated host matrix with data from this matrix.
The content of the provided host matrix will be overwritten; the 2 matrices must have the
same element type and dimension.

**Parameters**

| Name | Description |
| --- | --- |
| `hostMatrix` | the host-memory-backed matrix to fill. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:223`_

### toHost

```java
CuVSHostMatrix toHost()
```

Returns a host matrix; if the matrix is already a host matrix, a "weak" reference to the same host memory
is returned. If the matrix is a device matrix, a newly allocated matrix will be populated with data from
the device matrix.
The returned host matrix will need to be managed by the caller, which will be
responsible to call `CuVSMatrix#close()` to free its resources when done.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:232`_

### toDevice

```java
void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources)
```

Fills the provided, pre-allocated device matrix with data from this matrix.
The content of the provided device matrix will be overwritten; the 2 matrices must have the
same element type and dimension.

**Parameters**

| Name | Description |
| --- | --- |
| `deviceMatrix` | the device-memory-backed matrix to fill. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:241`_

### toDevice

```java
CuVSDeviceMatrix toDevice(CuVSResources cuVSResources)
```

Returns a device matrix; if this matrix is already a device matrix, a "weak" reference to the same host memory
is returned. If the matrix is a host matrix, a newly allocated matrix will be populated with data from
the host matrix.
The returned device matrix will need to be managed by the caller, which will be
responsible to call `CuVSMatrix#close()` to free its resources when done.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:250`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSMatrix.java:17`_
