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
static CuVSMatrix createFloatMatrix(List<float[]> data, int dimensions, CuVSResources resources)
```

A method to build a CuVSMatrix from a list of float vectors.

Uses CuVSMatrix.Builder to copy vectors directly to device memory
without creating intermediate heap arrays.

**Parameters**

| Name | Description |
| --- | --- |
| `data` | The float vectors |
| `dimensions` | The number float elements in each vector |
| `resources` | The CuVS resources for device matrix creation |

**Returns**

an instance of CuVSMatrix

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:63`_

### createByteMatrix

```java
static CuVSMatrix createByteMatrix( List<byte[]> data, int bytesPerVector, CuVSResources resources)
```

A method to build a CuVSMatrix from a list of byte vectors (for binary quantized vectors).

Uses CuVSMatrix.Builder to copy vectors directly to device memory
without creating intermediate heap arrays.

**Parameters**

| Name | Description |
| --- | --- |
| `data` | The byte vectors (packed bits for binary quantization) |
| `bytesPerVector` | The number of bytes in each vector |
| `resources` | The CuVS resources for device matrix creation |

**Returns**

an instance of CuVSMatrix with BYTE data type

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:92`_

### createByteMatrixFromArray

```java
static CuVSMatrix createByteMatrixFromArray( byte[][] data, int bytesPerVector, CuVSResources resources)
```

A method to build a CuVSMatrix from a 2D byte array (for binary quantized vectors).

**Parameters**

| Name | Description |
| --- | --- |
| `data` | The 2D byte array (packed bits for binary quantization) |
| `bytesPerVector` | The number of bytes in each vector |
| `resources` | The CuVS resources for device matrix creation |

**Returns**

an instance of CuVSMatrix with BYTE data type

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:119`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:141`_

### cuVSResourcesOrNull

```java
static CuVSResources cuVSResourcesOrNull()
```

Creates an instance of CuVSResources.

**Returns**

an instance of CuVSResources

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:150`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:178`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:192`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:210`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:26`_
