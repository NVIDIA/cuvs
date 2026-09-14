---
slug: api-reference/java-api-com-nvidia-cuvs-filterbitsethandle
---

# FilterBitsetHandle

_Java package: `com.nvidia.cuvs`_

```java
public interface FilterBitsetHandle extends AutoCloseable
```

Holds a precomputed multi-partition filter bitset and manages its device-memory lifecycle.

The packed `long[]` host arrays are immutable after construction. A single shared device
allocation is uploaded lazily on first use and reused thereafter.

Device pool configuration

Filter bitset device allocations use a shared, process-lifetime resources object with a
growable RMM pool initially sized to 4 MiB. Applications can set the
`com.nvidia.cuvs.filterBitsetPoolSize` system property before the first filter bitset upload
to customize it: zero explicitly disables pooling, while a positive value selects the initial
size. An invalid or negative value produces a warning and uses the 4 MiB default.

The initial size is a reservation, not a memory cap or a host-side cache policy. The pool can
grow as needed.

Lifecycle

The handle is reference-counted. Construction grants one initial reference, held by the owner
(typically a host-level cache), which is released by `#close()`. A thread that uses the
handle concurrently — e.g. while it may be evicted and closed by another thread — must guard the
use with `#tryIncRef()` / `#decRef()`. The shared device allocation is released only
when the last reference is dropped, so a concurrent `#close()` cannot free memory that is
still in use.

## Public Members

### tryIncRef

```java
boolean tryIncRef()
```

Attempts to acquire a reference to this handle, preventing its device allocation from being
released until a matching `#decRef()`. Callers that pass the handle to a search (or
otherwise touch its device allocation) must hold a reference for the duration of that use.

**Returns**

`true` if a reference was acquired; `false` if the handle has already been fully released, in which case no reference is acquired and it must not be used

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/FilterBitsetHandle.java:46`_

### decRef

```java
void decRef()
```

Releases a reference previously acquired via `#tryIncRef()`. When the last outstanding
reference is released, the shared device allocation is freed.

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if called without a matching `#tryIncRef()` |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/FilterBitsetHandle.java:54`_

### create

```java
static FilterBitsetHandle create(long[] combinedLongs)
```

Creates a handle from one partition's pre-packed bitset (one bit per vector in that partition).
In a multi-partition search each partition supplies its own handle.

**Parameters**

| Name | Description |
| --- | --- |
| `combinedLongs` | packed bitset words for a single partition |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/FilterBitsetHandle.java:62`_

### close

```java
@Override void close()
```

Releases the initial reference held since construction. Equivalent to a single `#decRef()`
of the owner's reference; the device allocation is freed once this and every reference acquired
via `#tryIncRef()` has been released. Idempotent — releasing the initial reference more
than once has no effect.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/FilterBitsetHandle.java:72`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/FilterBitsetHandle.java:37`_
