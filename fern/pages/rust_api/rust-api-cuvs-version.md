---
slug: api-reference/rust-api-cuvs-version
---

# Version Module

_Rust module: `cuvs::version`_

_Source: `rust/cuvs/src/version.rs`_

cuVS library version query.

## version

```rust
pub fn version() -> Result<(u16, u16, u16), LibraryError>
```

Returns the cuVS library version as `(major, minor, patch)`.

_Source: `rust/cuvs/src/version.rs:12`_
