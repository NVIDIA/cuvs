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

_Source: `rust/cuvs/src/lib.rs:13`_

## dlpack

```rust
pub mod dlpack;
```

_Source: `rust/cuvs/src/lib.rs:14`_

## neighbors

```rust
pub mod neighbors;
```

_Source: `rust/cuvs/src/lib.rs:16`_

## dlpack::\{AsDlTensor, AsDlTensorMut, DLPackError, DLTensorView, DLTensorViewMut, DType\}

```rust
pub use dlpack::{AsDlTensor, AsDlTensorMut, DLPackError, DLTensorView, DLTensorViewMut, DType};
```

_Source: `rust/cuvs/src/lib.rs:21`_

## error::LibraryError

```rust
pub use error::LibraryError;
```

_Source: `rust/cuvs/src/lib.rs:22`_

## resources::Resources

```rust
pub use resources::Resources;
```

_Source: `rust/cuvs/src/lib.rs:23`_
