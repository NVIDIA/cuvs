---
slug: api-reference/rust-api-cuvs-neighbors-vamana
---

# Neighbors Vamana Module

_Rust module: `cuvs::neighbors::vamana`_

_Source: `rust/cuvs/src/neighbors/vamana/mod.rs`_

Vamana: builds a DiskANN-style Vamana graph over a dataset.

Build an [`Index`] from a dataset (then typically serialize it). The dataset
is borrowed through the `AsDlTensor` trait; see the [`dlpack`](crate::dlpack)
module for the tensor model.

## index::Index

```rust
pub use index::Index;
```

_Source: `rust/cuvs/src/neighbors/vamana/mod.rs:14`_

## params::IndexParams

```rust
pub use params::IndexParams;
```

_Source: `rust/cuvs/src/neighbors/vamana/mod.rs:15`_

## VamanaError

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VamanaError {
    /* variants omitted */
}
```

Error type for Vamana operations.

_Source: `rust/cuvs/src/neighbors/vamana/mod.rs:23`_
