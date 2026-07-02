---
slug: api-reference/rust-api-cuvs-dlpack
---

# Dlpack Module

_Rust module: `cuvs::dlpack`_

_Source: `rust/cuvs/src/dlpack.rs`_

## ManagedTensor

```rust
#[derive(Debug)]
pub struct ManagedTensor(ffi::DLManagedTensor); {
    /* private fields */
}
```

ManagedTensor is a wrapper around a dlpack DLManagedTensor object.
This lets you pass matrices in device or host memory into cuvs.

**Methods**

| Name | Source |
| --- | --- |
| `as_ptr` | `rust/cuvs/src/dlpack.rs:21` |
| `to_device` | `rust/cuvs/src/dlpack.rs:27` |
| `to_host` | `rust/cuvs/src/dlpack.rs:47` |

### as_ptr

```rust
pub fn as_ptr(&self) -> *mut ffi::DLManagedTensor
```

_Source: `rust/cuvs/src/dlpack.rs:21`_

### to_device

```rust
pub fn to_device(&self, res: &Resources) -> Result<ManagedTensor>
```

Creates a new ManagedTensor on the current GPU device, and copies
the data into it.

_Source: `rust/cuvs/src/dlpack.rs:27`_

### to_host

```rust
pub fn to_host<
T: IntoDtype,
S: ndarray::RawData<Elem = T> + ndarray::RawDataMut,
D: ndarray::Dimension,
>(
&self,
res: &Resources,
arr: &mut ndarray::ArrayBase<S, D>,
) -> Result<()>
```

Copies data from device memory into host memory

_Source: `rust/cuvs/src/dlpack.rs:47`_

_Source: `rust/cuvs/src/dlpack.rs:14`_

## IntoDtype

```rust
pub trait IntoDtype {
    /* required methods omitted */
}
```

_Source: `rust/cuvs/src/dlpack.rs:16`_
