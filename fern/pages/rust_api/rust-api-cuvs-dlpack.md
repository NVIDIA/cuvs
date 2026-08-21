---
slug: api-reference/rust-api-cuvs-dlpack
---

# Dlpack Module

_Rust module: `cuvs::dlpack`_

_Source: `rust/cuvs/src/dlpack.rs`_

DLPack tensor interop.

cuVS exchanges tensors with the C library through the DLPack ABI. This crate
never owns tensor storage: every entry point borrows a value that exposes a
view through the [`AsDlTensor`] / [`AsDlTensorMut`] traits:

* [`DLTensorView`] — a read-only view, for inputs the C API only reads
(datasets, queries).
* [`DLTensorViewMut`] — a writable view, for outputs the C API writes
(neighbors, distances).

A view is non-owning. The traits take `&self` / `&mut self`, so the compiler
ties the view's lifetime to the borrow of the value that owns the underlying
buffer. The view materializes a stack-local [`DLManagedTensor`] only for the
duration of each FFI call.

The crate ships no public tensor type. To hand your own GPU (or host) buffer
to cuVS, implement [`AsDlTensor`] / [`AsDlTensorMut`] for it on top of
[`DLTensorView::from_raw_parts`]. Most algorithms require search inputs and
outputs to live in device memory. See `examples/cagra.rs` for a complete,
runnable adapter built on the raw CUDA runtime.

## ffi::\{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor\}

```rust
pub use ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor};
```

_Source: `rust/cuvs/src/dlpack.rs:34`_

## AsDlTensor

```rust
pub trait AsDlTensor {
    /* required methods omitted */
}
```

Borrows a tensor as a read-only [`DLTensorView`] for tensor inputs.

Implement this for your own tensor type by calling
[`DLTensorView::from_raw_parts`] inside a small `unsafe` block, upholding its
safety contract.

#### Examples

A minimal adapter for a row-major matrix in device memory:

```
use cuvs::dlpack::{AsDlTensor, DLDevice, DLDeviceType, DLPackError, DLTensorView, DType};

struct GpuMatrix<T> {
ptr: *mut T,
rows: usize,
cols: usize,
}

impl<T: DType> AsDlTensor for GpuMatrix<T> {
fn as_dl_tensor(&self) -> Result<DLTensorView<'_>, DLPackError> {
let shape = [self.rows as i64, self.cols as i64];
// SAFETY: `ptr` points to `rows * cols` initialized elements of `T`
// in device 0's memory, valid while `self` is borrowed, and is
// row-major contiguous.
unsafe {
DLTensorView::from_raw_parts(
self.ptr.cast(),
DLDevice { device_type: DLDeviceType::kDLCUDA, device_id: 0 },
&shape,
None,
T::dl_dtype(),
)
}
}
}
```

_Source: `rust/cuvs/src/dlpack.rs:79`_

## AsDlTensorMut

```rust
pub trait AsDlTensorMut {
    /* required methods omitted */
}
```

Borrows a tensor as a writable [`DLTensorViewMut`] for tensor outputs.

In addition to the [`DLTensorView::from_raw_parts`] invariants, writable
adapters must guarantee exclusive access to the data region. The `&mut self`
receiver makes the compiler enforce that exclusivity for the borrow.

_Source: `rust/cuvs/src/dlpack.rs:88`_

## DType

```rust
pub trait DType {
    /* required methods omitted */
}
```

Maps a Rust element type to a DLPack [`DLDataType`].

_Source: `rust/cuvs/src/dlpack.rs:93`_

## DLPackError

```rust
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum DLPackError {
    /* variants omitted */
}
```

Error when converting an external tensor to a DLPack view.

_Source: `rust/cuvs/src/dlpack.rs:121`_

## DLTensorView

```rust
#[must_use]
pub struct DLTensorView<'a> {
    /* private fields */
}
```

A non-owning, read-only DLPack tensor view.

**Methods**

| Name | Source |
| --- | --- |
| `from_raw_parts` | `rust/cuvs/src/dlpack.rs:176` |
| `ndim` | `rust/cuvs/src/dlpack.rs:226` |
| `shape` | `rust/cuvs/src/dlpack.rs:231` |
| `strides` | `rust/cuvs/src/dlpack.rs:236` |
| `dtype` | `rust/cuvs/src/dlpack.rs:241` |
| `device` | `rust/cuvs/src/dlpack.rs:246` |

### from_raw_parts

```rust
pub unsafe fn from_raw_parts(
data: *mut std::ffi::c_void,
device: ffi::DLDevice,
shape: &[i64],
strides: Option<&[i64]>,
dtype: ffi::DLDataType,
) -> std::result::Result<Self, DLPackError>
```

Construct a DLPack view from raw tensor metadata.

#### Safety

The caller must guarantee that:
- `data` points to initialized storage matching `shape`, `strides`, and
`dtype`, residing on the device described by `device`;
- that storage remains valid for the lifetime `'a`;
- the C API consumes the resulting [`DLManagedTensor`] (including its
`shape`/`strides` pointers) only for the duration of the FFI call and
does not retain it afterward — cuVS upholds this.

_Source: `rust/cuvs/src/dlpack.rs:176`_

### ndim

```rust
pub fn ndim(&self) -> usize
```

Number of dimensions.

_Source: `rust/cuvs/src/dlpack.rs:226`_

### shape

```rust
pub fn shape(&self) -> &[i64]
```

Shape of the tensor.

_Source: `rust/cuvs/src/dlpack.rs:231`_

### strides

```rust
pub fn strides(&self) -> Option<&[i64]>
```

Strides, if non-contiguous. `None` means row-major contiguous.

_Source: `rust/cuvs/src/dlpack.rs:236`_

### dtype

```rust
pub fn dtype(&self) -> ffi::DLDataType
```

Element data type.

_Source: `rust/cuvs/src/dlpack.rs:241`_

### device

```rust
pub fn device(&self) -> ffi::DLDevice
```

Device where the data resides.

_Source: `rust/cuvs/src/dlpack.rs:246`_

_Source: `rust/cuvs/src/dlpack.rs:155`_

## DLTensorViewMut

```rust
#[must_use]
pub struct DLTensorViewMut<'a> {
    /* private fields */
}
```

A non-owning, writable DLPack tensor view.

**Methods**

| Name | Source |
| --- | --- |
| `from_raw_parts` | `rust/cuvs/src/dlpack.rs:274` |

### from_raw_parts

```rust
pub unsafe fn from_raw_parts(
data: *mut std::ffi::c_void,
device: ffi::DLDevice,
shape: &[i64],
strides: Option<&[i64]>,
dtype: ffi::DLDataType,
) -> std::result::Result<Self, DLPackError>
```

Construct a writable DLPack view from raw tensor metadata.

#### Safety

In addition to the [`DLTensorView::from_raw_parts`] invariants, the
caller must guarantee the storage is exclusively writable for `'a`.

_Source: `rust/cuvs/src/dlpack.rs:274`_

_Source: `rust/cuvs/src/dlpack.rs:262`_
