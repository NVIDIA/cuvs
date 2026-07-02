---
slug: api-reference/rust-api-cuvs-error
---

# Error Module

_Rust module: `cuvs::error`_

_Source: `rust/cuvs/src/error.rs`_

Low-level error handling shared by every cuVS module.

[`check_cuvs`] turns a raw `cuvsError_t` status into a [`LibraryError`], which
each module's error type wraps via `#[from]`.

## LibraryError

```rust
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct LibraryError(Cow<'static, str>); {
    /* private fields */
}
```

A failure reported by the cuVS C library.

Carries the message captured from `cuvsGetLastErrorText` at the point of
failure. Every module's error type wraps this via `#[from]`.

_Source: `rust/cuvs/src/error.rs:19`_
