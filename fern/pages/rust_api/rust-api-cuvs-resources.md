---
slug: api-reference/rust-api-cuvs-resources
---

# Resources Module

_Rust module: `cuvs::resources`_

_Source: `rust/cuvs/src/resources.rs`_

GPU resource management with RAII semantics.

## ResourcesError

```rust
#[derive(Debug, thiserror::Error)]
pub enum ResourcesError {
    /* variants omitted */
}
```

Error type for resource operations.

_Source: `rust/cuvs/src/resources.rs:18`_

## Resources

```rust
#[derive(Debug)]
pub struct Resources {
    /* private fields */
}
```

Resources are objects that are shared between function calls,
and includes things like CUDA streams, cuBLAS handles and other
resources that are expensive to create.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/resources.rs:37` |
| `with_memory_tracking` | `rust/cuvs/src/resources.rs:54` |
| `with_stream` | `rust/cuvs/src/resources.rs:81` |
| `stream` | `rust/cuvs/src/resources.rs:90` |
| `sync_stream` | `rust/cuvs/src/resources.rs:99` |

### new

```rust
pub fn new() -> Result<Resources>
```

Creates a new resources handle bound to the current CUDA device.

_Source: `rust/cuvs/src/resources.rs:37`_

### with_memory_tracking

```rust
pub fn with_memory_tracking(
csv_path: impl AsRef<Path>,
sample_interval: Option<Duration>,
) -> Result<Resources>
```

Returns a new `Resources` object whose memory allocations are tracked
and written as CSV samples to `csv_path` from a background thread.

The handle wraps all reachable memory resources (host, pinned, managed,
device, workspace, large_workspace) with allocation-tracking adaptors
and replaces the global host and device memory resources for the
lifetime of the handle. The CSV reporter is stopped and the global
memory resources are restored when the handle is dropped.

`sample_interval` controls the minimum time between successive CSV
samples; when `None`, the C++ default of 10 ms is used.

_Source: `rust/cuvs/src/resources.rs:54`_

### with_stream

```rust
pub unsafe fn with_stream(stream: ffi::cudaStream_t) -> Result<Resources>
```

Creates a resources handle that enqueues work on `stream` instead of the
default internal stream.

The stream is bound once, at construction.

#### Safety

`stream` must be a valid CUDA stream for the current device and must
remain valid for as long as this handle uses it.

_Source: `rust/cuvs/src/resources.rs:81`_

### stream

```rust
pub fn stream(&self) -> Result<ffi::cudaStream_t>
```

Returns the current CUDA stream associated with this handle.

_Source: `rust/cuvs/src/resources.rs:90`_

### sync_stream

```rust
pub fn sync_stream(&self) -> Result<()>
```

Blocks until all operations on the current CUDA stream have completed.

_Source: `rust/cuvs/src/resources.rs:99`_

_Source: `rust/cuvs/src/resources.rs:31`_
