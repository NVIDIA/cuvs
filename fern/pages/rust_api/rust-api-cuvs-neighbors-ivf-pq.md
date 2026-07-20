---
slug: api-reference/rust-api-cuvs-neighbors-ivf-pq
---

# Neighbors Ivf Pq Module

_Rust module: `cuvs::neighbors::ivf_pq`_

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs`_

IVF-PQ: an inverted-file index that product-quantizes the vectors. Like
IVF-Flat it partitions the dataset into `n_lists` clusters and scans the
`n_probes` closest at query time, but compresses each vector into `pq_dim`
codes of `pq_bits` bits — much smaller, slightly less accurate.

Build an [`Index`] from a dataset, then [`search`](Index::search) it with
device-resident queries and output buffers. Tensors are borrowed through the
`AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
module for the tensor model and `examples/cagra.rs` for the same build/search
workflow.

## index::Index

```rust
pub use index::Index;
```

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:19`_

## params::\{IndexParams, SearchParams\}

```rust
pub use params::{IndexParams, SearchParams};
```

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:20`_

## CodebookGen

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodebookGen {
    /* variants omitted */
}
```

Strategy for creating PQ codebooks.

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:28`_

## ListLayout

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum ListLayout {
    /* variants omitted */
}
```

Memory layout of the IVF-PQ list data.

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:56`_

## LutDType

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum LutDType {
    /* variants omitted */
}
```

Lookup-table dtype used during IVF-PQ search.

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:84`_

## InternalDistanceDType

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum InternalDistanceDType {
    /* variants omitted */
}
```

Accumulator dtype used for internal IVF-PQ distance computation.

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:106`_

## IvfPqError

```rust
#[derive(Debug, thiserror::Error)]
pub enum IvfPqError {
    /* variants omitted */
}
```

Error type for IVF-PQ operations.

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:124`_
