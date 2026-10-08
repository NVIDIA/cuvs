---
slug: api-reference/rust-api-cuvs
---

# cuVS Rust Crate

_Rust module: `cuvs`_

_Source: `rust/cuvs/src/lib.rs`_

cuVS: Rust bindings for Vector Search on the GPU

This crate provides Rust bindings for cuVS, allowing you to run
approximate nearest neighbors search on the GPU.

## cluster

```rust
pub mod cluster;
```

_Source: `rust/cuvs/src/lib.rs:12`_

## distance

```rust
pub mod distance;
```

_Source: `rust/cuvs/src/lib.rs:14`_

## dlpack

```rust
pub mod dlpack;
```

_Source: `rust/cuvs/src/lib.rs:15`_

## error

```rust
pub mod error;
```

_Source: `rust/cuvs/src/lib.rs:16`_

## neighbors

```rust
pub mod neighbors;
```

_Source: `rust/cuvs/src/lib.rs:18`_

## resources

```rust
pub mod resources;
```

_Source: `rust/cuvs/src/lib.rs:19`_

## version

```rust
pub mod version;
```

_Source: `rust/cuvs/src/lib.rs:20`_

## dlpack::\{AsDlTensor, AsDlTensorMut, DLPackError, DLTensorView, DLTensorViewMut, DType\}

```rust
pub use dlpack::{AsDlTensor, AsDlTensorMut, DLPackError, DLTensorView, DLTensorViewMut, DType};
```

_Source: `rust/cuvs/src/lib.rs:25`_

## error::LibraryError

```rust
pub use error::LibraryError;
```

_Source: `rust/cuvs/src/lib.rs:26`_

## resources::Resources

```rust
pub use resources::Resources;
```

_Source: `rust/cuvs/src/lib.rs:27`_

## ReadmeDocTests

```rust
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct ReadmeDocTests; {
    /* private fields */
}
```

_Source: `rust/cuvs/src/lib.rs:33`_
