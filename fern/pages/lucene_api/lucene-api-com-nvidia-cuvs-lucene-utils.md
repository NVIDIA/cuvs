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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:39`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:59`_

### createHostFloatMatrix

```java
static CuVSHostMatrix createHostFloatMatrix(List<float[]> data, int dimensions)
```

Builds a host-memory CuVSMatrix from a list of float vectors.

Copies vectors directly into native host memory without creating an intermediate \{@code
float[][]\} on the heap.

**Parameters**

| Name | Description |
| --- | --- |
| `data` | The float vectors |
| `dimensions` | The number of float elements in each vector |

**Returns**

a host-memory CuVSMatrix

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:87`_

### createHostByteMatrix

```java
static CuVSHostMatrix createHostByteMatrix(List<byte[]> data, int bytesPerVector)
```

Builds a host-memory CuVSMatrix from byte vectors without first materializing the list as an
intermediate `byte[][]`.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:101`_

### createHostByteMatrixFromArray

```java
static CuVSHostMatrix createHostByteMatrixFromArray(byte[][] data, int bytesPerVector)
```

Builds a host-memory CuVSMatrix from a 2D byte array.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:112`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:128`_

### cuVSResourcesOrNull

```java
static CuVSResources cuVSResourcesOrNull()
```

Creates an instance of CuVSResources.

**Returns**

an instance of CuVSResources

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:137`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:165`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:179`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:22`_
