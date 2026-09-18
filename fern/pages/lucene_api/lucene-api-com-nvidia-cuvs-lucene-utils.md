---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-utils
---

# Utils

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class Utils
```

This class provides common static utility methods.

## Public Members

### handleThrowable

```java
static RuntimeException handleThrowable(Throwable t) throws IOException
```

A utility method that rethrows known throwable types without changing their identity.

In particular, `Error` instances must not be converted to a \{@link
RuntimeException\}; callers rely on errors retaining their original type and stack trace.

This method never returns normally; its return type exists solely so callers can write
`throw handleThrowable(t);`, letting the compiler verify that the enclosing statement
always completes abruptly.

**Parameters**

| Name | Description |
| --- | --- |
| `t` | the throwable object |

**Returns**

never returns; always throws

**Throws**

| Type | Description |
| --- | --- |
| `IOException` |  |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:43`_

### createFloatMatrix

```java
static CuVSMatrix createFloatMatrix(List<float[]> data, int dimensions) throws IOException
```

Builds a host-memory CuVSMatrix from a list of float vectors.

Copies vectors directly into a native host matrix via `CuVSMatrix#hostBuilder`,
without creating an intermediate `float[][]` on the heap.

**Parameters**

| Name | Description |
| --- | --- |
| `data` | The float vectors |
| `dimensions` | The number of float elements in each vector |

**Returns**

a host-memory CuVSMatrix

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:62`_

### createByteMatrix

```java
static CuVSMatrix createByteMatrix(List<byte[]> data, int bytesPerVector) throws IOException
```

Builds a host-memory CuVSMatrix from a list of byte vectors (e.g. quantized vectors).

**Parameters**

| Name | Description |
| --- | --- |
| `data` | The byte vectors (packed bits for binary quantization) |
| `bytesPerVector` | The number of bytes in each vector |

**Returns**

a host-memory CuVSMatrix with BYTE data type

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:80`_

### createByteMatrixFromArray

```java
static CuVSMatrix createByteMatrixFromArray(byte[][] data, int bytesPerVector) throws IOException
```

Builds a host-memory CuVSMatrix from a 2D byte array (e.g. quantized vectors).

**Parameters**

| Name | Description |
| --- | --- |
| `data` | The 2D byte array (packed bits for binary quantization) |
| `bytesPerVector` | The number of bytes in each vector |

**Returns**

a host-memory CuVSMatrix with BYTE data type

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:98`_

### closeIndexWithDatasetFallback

```java
static void closeIndexWithDatasetFallback(AutoCloseable index, AutoCloseable dataset) throws Exception
```

Closes an index that owns `dataset`. If index cleanup fails before releasing the
dataset, a direct dataset close is attempted and attached to the index failure when needed.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:114`_

### ownDataset

```java
static <I extends AutoCloseable> OwnedIndex<I> ownDataset(AutoCloseable dataset)
```

Starts an ownership scope for a dataset that may later be transferred to an index.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:131`_

### nanosToMillis

```java
static long nanosToMillis(long nanos)
```

A utility method to convert nanoseconds to milliseconds.

**Parameters**

| Name | Description |
| --- | --- |
| `nanos` |  |

**Returns**

milliseconds

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:196`_

### cuVSResourcesOrNull

```java
static CuVSResources cuVSResourcesOrNull()
```

Creates an instance of CuVSResources.

**Returns**

an instance of CuVSResources

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:205`_

### handleThrowableWithIgnore

```java
static void handleThrowableWithIgnore(Throwable t, String msg) throws IOException
```

A utility method that conditionally ignores certain throwable objects

**Parameters**

| Name | Description |
| --- | --- |
| `t` | the throwable object |
| `msg` | the message to check |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` |  |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:233`_

### createListFromMergedVectors

```java
static List<float[]> createListFromMergedVectors(FloatVectorValues mergedVectorValues) throws IOException
```

Creates a list of float vectors from the input

**Parameters**

| Name | Description |
| --- | --- |
| `mergedVectorValues` | instance of `FloatVectorValues` |

**Returns**

a list of float arrays

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exception |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:247`_

### info

```java
static void info(InfoStream infoStream, String component, String msg)
```

Utility to print info/debug messages via InfoStream.

**Parameters**

| Name | Description |
| --- | --- |
| `infoStream` | the writer's infostream |
| `component` | the name of the index writer |
| `msg` | the log message to push via the InfoStream |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:265`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:26`_
