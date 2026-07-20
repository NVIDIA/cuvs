---
slug: api-reference/rust-api-cuvs-neighbors-ivf-flat
---

# Neighbors Ivf Flat Module

_Rust module: `cuvs::neighbors::ivf_flat`_

_Source: `rust/cuvs/src/neighbors/ivf_flat/mod.rs`_

IVF-Flat: an inverted-file index over uncompressed ("flat") vectors. It
partitions the dataset into `n_lists` clusters and, at query time, scans only
the `n_probes` closest clusters — a simple knob to trade recall for speed.

Build an [`Index`] from a dataset, then [`search`](Index::search) it with
device-resident queries and output buffers. Tensors are borrowed through the
`AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
module for the tensor model and `examples/cagra.rs` for the same build/search
workflow.

## crate::neighbors::filters::SearchFilter

```rust
pub use crate::neighbors::filters::SearchFilter;
```

_Source: `rust/cuvs/src/neighbors/ivf_flat/mod.rs:19`_

## index::Index

```rust
pub use index::Index;
```

_Source: `rust/cuvs/src/neighbors/ivf_flat/mod.rs:20`_

## params::\{IndexParams, SearchParams\}

```rust
pub use params::{IndexParams, SearchParams};
```

_Source: `rust/cuvs/src/neighbors/ivf_flat/mod.rs:21`_

## IvfFlatError

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IvfFlatError {
    /* variants omitted */
}
```

Error type for IVF-Flat operations.

_Source: `rust/cuvs/src/neighbors/ivf_flat/mod.rs:29`_
