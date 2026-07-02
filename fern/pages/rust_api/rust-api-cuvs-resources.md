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

_Source: `rust/cuvs/src/resources.rs:15`_

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
| `new` | `rust/cuvs/src/resources.rs:31` |
| `with_stream` | `rust/cuvs/src/resources.rs:46` |
| `stream` | `rust/cuvs/src/resources.rs:55` |
| `sync_stream` | `rust/cuvs/src/resources.rs:64` |

### new

```rust
pub fn new() -> Result<Resources>
```

Creates a new resources handle bound to the current CUDA device.

_Source: `rust/cuvs/src/resources.rs:31`_

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

_Source: `rust/cuvs/src/resources.rs:46`_

### stream

```rust
pub fn stream(&self) -> Result<ffi::cudaStream_t>
```

Returns the current CUDA stream associated with this handle.

_Source: `rust/cuvs/src/resources.rs:55`_

### sync_stream

```rust
pub fn sync_stream(&self) -> Result<()>
```

Blocks until all operations on the current CUDA stream have completed.

_Source: `rust/cuvs/src/resources.rs:64`_

_Source: `rust/cuvs/src/resources.rs:25`_
