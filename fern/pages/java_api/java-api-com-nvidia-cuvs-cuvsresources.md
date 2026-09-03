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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:26`_

### access

```java
ScopedAccess access()
```

Gets scoped access to the native resources object.
The native resource object is not thread safe: only a single thread at every time should access
concurrently the same native resources. Calling this method from multiple thread is OK, but the
returned `ScopedAccess` object must be closed before calling `access()` again from a
different thread.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:39`_

### deviceId

```java
int deviceId()
```

Get the logical id of the device associated with this resources object.
Information about the device id is immutable, so it is safe to expose it without getting `ScopedAccess`
to the enclosing resources.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:46`_

### close

```java
@Override void close()
```

Closes this CuVSResources object and releases any resources associated with it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:51`_

### tempDirectory

```java
Path tempDirectory()
```

The temporary directory to use for intermediate operations.
Defaults to \{@systemProperty java.io.tmpdir\}.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:58`_

### setWorkspacePool

```java
void setWorkspacePool(long initialSizeBytes)
```

Configure the temporary workspace on this resources object as an uncapped pool backed by the
current device memory resource. After the initial reservation is allocated on first use,
subsequent calls to `cuvsRMMAlloc` / `cuvsRMMFree` on this handle hit the pool
cache rather than calling `cudaMallocAsync` / `cudaFreeAsync`, reducing CUDA
context lock contention under concurrent query threads. The pool grows without shrinking:
freed allocations are returned to the pool rather than to the device, so the pool's
high-water mark only increases until the resources object is closed.

The pool is per-resources-handle (i.e. per query thread when resources are thread-local),
so there is no cross-thread pool mutex contention. Call this once after creating the resources
object; calling it again replaces the pool.

**Parameters**

| Name | Description |
| --- | --- |
| `initialSizeBytes` | initial pool reservation in bytes; must be `&gt; 0`. Size `initialSizeBytes` to cover the steady-state working set to avoid growth after warmup |

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if `initialSizeBytes` is not greater than 0 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:78`_

### create

```java
static CuVSResources create() throws Throwable
```

Creates a new resources.
Equivalent to
\{@code
create(CuVSProvider.tempDirectory())
\}

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:87`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:98`_

### create

```java
static CuVSResources create( Path tempDirectory, Path memoryTrackingCsvPath, Duration memoryTrackingSampleInterval) throws Throwable
```

Creates a new resources whose memory allocations are tracked and written as
CSV samples from a background thread.

The returned handle wraps all reachable memory resources (host, pinned,
managed, device, workspace, large_workspace) with allocation-tracking
adaptors and replaces the global host and device memory resources for the
lifetime of the handle. It is otherwise indistinguishable from a handle
created by `#create(Path)` and can be used wherever a
`CuVSResources` is accepted. The CSV reporter is stopped and the
global memory resources are restored when the handle is closed.

**Parameters**

| Name | Description |
| --- | --- |
| `tempDirectory` | the temporary directory to use for intermediate operations |
| `memoryTrackingCsvPath` | path to the output CSV file (created/truncated) |
| `memoryTrackingSampleInterval` | minimum interval between successive CSV samples |

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | if the provider does not support cuvs |
| `LibraryException` | if the native library cannot be loaded |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:123`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResources.java:16`_
