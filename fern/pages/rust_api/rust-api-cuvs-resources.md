---
slug: api-reference/rust-api-cuvs-resources
---

# Resources Module

_Rust module: `cuvs::resources`_

_Source: `rust/cuvs/src/resources.rs`_

## Resources

```rust
#[derive(Debug)]
pub struct Resources(pub ffi::cuvsResources_t); {
    /* private fields */
}
```

Resources are objects that are shared between function calls,
and includes things like CUDA streams, cuBLAS handles and other
resources that are expensive to create.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/resources.rs:17` |
| `set_cuda_stream` | `rust/cuvs/src/resources.rs:26` |
| `get_cuda_stream` | `rust/cuvs/src/resources.rs:31` |
| `sync_stream` | `rust/cuvs/src/resources.rs:40` |

### new

```rust
pub fn new() -> Result<Resources>
```

Returns a new Resources object

_Source: `rust/cuvs/src/resources.rs:17`_

### set_cuda_stream

```rust
pub fn set_cuda_stream(&self, stream: ffi::cudaStream_t) -> Result<()>
```

Sets the current cuda stream

_Source: `rust/cuvs/src/resources.rs:26`_

### get_cuda_stream

```rust
pub fn get_cuda_stream(&self) -> Result<ffi::cudaStream_t>
```

Gets the current cuda stream

_Source: `rust/cuvs/src/resources.rs:31`_

### sync_stream

```rust
pub fn sync_stream(&self) -> Result<()>
```

Syncs the current cuda stream

_Source: `rust/cuvs/src/resources.rs:40`_

_Source: `rust/cuvs/src/resources.rs:13`_
