---
slug: api-reference/rust-api-cuvs-error
---

# Error Module

_Rust module: `cuvs::error`_

_Source: `rust/cuvs/src/error.rs`_

## CuvsError

```rust
#[derive(Debug, Clone)]
pub struct CuvsError {
    /* private fields */
}
```

_Source: `rust/cuvs/src/error.rs:9`_

## Error

```rust
#[derive(Debug, Clone)]
pub enum Error {
    /* variants omitted */
}
```

_Source: `rust/cuvs/src/error.rs:15`_

## Result

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

_Source: `rust/cuvs/src/error.rs:25`_

## check_cuvs

```rust
pub fn check_cuvs(err: ffi::cuvsError_t) -> Result<()>
```

Simple wrapper to convert a cuvsError_t into a Result

_Source: `rust/cuvs/src/error.rs:43`_
