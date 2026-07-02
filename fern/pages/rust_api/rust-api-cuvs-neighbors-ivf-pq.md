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

## params::\{ IndexParams, SearchParams, cudaDataType_t, cuvsIvfPqCodebookGen, cuvsIvfPqListLayout, \}

```rust
pub use params::{
IndexParams, SearchParams, cudaDataType_t, cuvsIvfPqCodebookGen, cuvsIvfPqListLayout,
};
```

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:20`_

## IvfPqError

```rust
#[derive(Debug, thiserror::Error)]
pub enum IvfPqError {
    /* variants omitted */
}
```

Error type for IVF-PQ operations.

_Source: `rust/cuvs/src/neighbors/ivf_pq/mod.rs:29`_
