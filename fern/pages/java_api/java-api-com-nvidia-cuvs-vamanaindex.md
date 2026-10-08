---
slug: api-reference/java-api-com-nvidia-cuvs-vamanaindex
---

# VamanaIndex

_Java package: `com.nvidia.cuvs`_

```java
public interface VamanaIndex extends AutoCloseable
```

`VamanaIndex` encapsulates a Vamana index, along with methods to build
it on the GPU and serialize it in the DiskANN file format.

Vamana is the graph construction algorithm behind DiskANN. cuVS currently
provides build and serialize only. There is no Vamana search API, so a
serialized index is searched by loading it with DiskANN.

## Public Members

### getDimensions

```java
int getDimensions() throws Throwable
```

Gets the dimensionality of the vectors in this index.

**Returns**

the number of dimensions

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:30`_

### serialize

```java
default void serialize(Path filePrefix) throws Throwable
```

Serializes the index in the DiskANN file format, including the dataset.

This writes two files, `filePrefix` holding the graph and
`filePrefix + ".data"` holding the dataset.

**Parameters**

| Name | Description |
| --- | --- |
| `filePrefix` | the prefix that output file names are derived from |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:40`_

### serialize

```java
void serialize(Path filePrefix, boolean includeDataset) throws Throwable
```

Serializes the index in the DiskANN file format.

When `includeDataset` is true this writes `filePrefix` holding
the graph and `filePrefix + ".data"` holding the dataset. When it is
false only `filePrefix` is written.

The argument is a prefix and not a complete file name, matching the native
`file_prefix` parameter.

**Parameters**

| Name | Description |
| --- | --- |
| `filePrefix` | the prefix that output file names are derived from |
| `includeDataset` | whether to write the dataset alongside the graph |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:57`_

### getCuVSResources

```java
CuVSResources getCuVSResources()
```

Gets an instance of `CuVSResources`

**Returns**

an instance of `CuVSResources`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:64`_

### newBuilder

```java
static Builder newBuilder(CuVSResources cuvsResources)
```

Creates a new Builder with an instance of `CuVSResources`.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsResources` | an instance of `CuVSResources` |

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | if the provider does not support cuvs |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:72`_

### withDataset

```java
Builder withDataset(float[][] vectors)
```

Sets the dataset for building the `VamanaIndex`.

**Parameters**

| Name | Description |
| --- | --- |
| `vectors` | a two-dimensional float array |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:88`_

### withDataset

```java
Builder withDataset(CuVSMatrix dataset)
```

Sets the dataset for building the `VamanaIndex`.

The native builder accepts `float`, `half`, `uint8`,
and `int8` datasets. Of those, `CuVSMatrix.DataType#FLOAT`,
`CuVSMatrix.DataType#HALF`, and `CuVSMatrix.DataType#BYTE`
are reachable from Java today, where `BYTE` is unsigned.
`int8` has no corresponding `DataType`.

The native index may retain a non-owning device view of the dataset
rather than copying it, so the caller must keep this matrix open for at
least as long as the index and close it afterwards. A dataset supplied as
a `float[][]` is created and closed by the index instead.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | a `CuVSMatrix` object containing the vectors |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:107`_

### withIndexParams

```java
Builder withIndexParams(VamanaIndexParams vamanaIndexParameters)
```

Registers an instance of configured `VamanaIndexParams` with this
Builder.

**Parameters**

| Name | Description |
| --- | --- |
| `vamanaIndexParameters` | An instance of VamanaIndexParams |

**Returns**

An instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:116`_

### build

```java
VamanaIndex build() throws Throwable
```

Builds and returns an instance of `VamanaIndex`.

**Returns**

an instance of `VamanaIndex`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:123`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndex.java:21`_
