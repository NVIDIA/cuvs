---
slug: api-reference/java-api-com-nvidia-cuvs-cuvsresources
---

# CuVSResources

_Java package: `com.nvidia.cuvs`_

```java
public interface CuVSResources extends AutoCloseable
```

Used for allocating resources for cuVS

## Public Members

### handle

```java
long handle()
```

Gets the opaque CuVSResources handle, to be used whenever we need to pass a cuvsResources_t parameter

**Returns**

the CuVSResources handle

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:25`_

### access

```java
ScopedAccess access()
```

Gets scoped access to the native resources object.
The native resource object is not thread safe: only a single thread at every time should access
concurrently the same native resources. Calling this method from multiple thread is OK, but the
returned `ScopedAccess` object must be closed before calling `access()` again from a
different thread.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:38`_

### deviceId

```java
int deviceId()
```

Get the logical id of the device associated with this resources object.
Information about the device id is immutable, so it is safe to expose it without getting `ScopedAccess`
to the enclosing resources.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:45`_

### close

```java
@Override void close()
```

Closes this CuVSResources object and releases any resources associated with it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:50`_

### tempDirectory

```java
Path tempDirectory()
```

The temporary directory to use for intermediate operations.
Defaults to \{@systemProperty java.io.tmpdir\}.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:57`_

### create

```java
static CuVSResources create() throws Throwable
```

Creates a new resources.
Equivalent to
\{@code
create(CuVSProvider.tempDirectory())
\}

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:66`_

### create

```java
static CuVSResources create(Path tempDirectory) throws Throwable
```

Creates a new resources.

**Parameters**

| Name | Description |
| --- | --- |
| `tempDirectory` | the temporary directory to use for intermediate operations |

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | if the provider does not cuvs |
| `LibraryException` | if the native library cannot be loaded |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:77`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:15`_
