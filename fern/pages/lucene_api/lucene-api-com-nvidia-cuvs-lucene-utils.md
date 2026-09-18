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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:40`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:159`_

### cuVSResourcesOrNull

```java
static CuVSResources cuVSResourcesOrNull()
```

Creates an instance of CuVSResources.

**Returns**

an instance of CuVSResources

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:168`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:196`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/Utils.java:23`_
