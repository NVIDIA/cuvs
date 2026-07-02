---
slug: api-reference/rust-api-cuvs-neighbors-cagra
---

# Neighbors Cagra Module

_Rust module: `cuvs::neighbors::cagra`_

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs`_

CAGRA: a graph-based approximate nearest neighbors algorithm with
state-of-the-art query throughput for both small and large batch sizes.

Build an [`Index`] from a dataset, then [`search`](Index::search) it with
device-resident queries and output buffers. Tensors are passed through the
`AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
module for the tensor model and `examples/cagra.rs` for a complete, runnable
example.

Parameter types ([`IndexParams`], [`SearchParams`], ...) use the [`bon`]
builder pattern: every setter is optional and unset values keep the cuVS C
library defaults. Values are validated when the builder's `build()` runs,
returning [`CagraError::Validation`] for out-of-range inputs.

## index::Index

```rust
pub use index::Index;
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:23`_

## params::\{CompressionParams, IndexParams, SearchParams\}

```rust
pub use params::{CompressionParams, IndexParams, SearchParams};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:24`_

## GraphBuildAlgo

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphBuildAlgo {
    /* variants omitted */
}
```

Algorithm for building the internal k-NN graph.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:32`_

## SearchAlgo

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchAlgo {
    /* variants omitted */
}
```

Search kernel implementation.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:72`_

## HashMode

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum HashMode {
    /* variants omitted */
}
```

Hash-table mode used during search.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:108`_

## CagraError

```rust
#[derive(Debug, thiserror::Error)]
pub enum CagraError {
    /* variants omitted */
}
```

Error type for CAGRA operations.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:139`_
