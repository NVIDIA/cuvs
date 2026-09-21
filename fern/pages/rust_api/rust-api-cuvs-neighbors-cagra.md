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

## crate::dataset::\{ CuvsDataset, Dataset, DatasetKind, DatasetView, PaddedDataset, PqDataset, \}

```rust
pub use crate::dataset::{
CuvsDataset, Dataset, DatasetKind, DatasetView, PaddedDataset, PqDataset,
};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:23`_

## crate::neighbors::filters::\{Bitset, Filter\}

```rust
pub use crate::neighbors::filters::{Bitset, Filter};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:26`_

## index::\{DeserializedIndex, Index\}

```rust
pub use index::{DeserializedIndex, Index};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:27`_

## params::\{CompressionParams, IndexParams, ProductQuantizerParams, SearchParams\}

```rust
pub use params::{CompressionParams, IndexParams, ProductQuantizerParams, SearchParams};
```

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:28`_

## make_pq_dataset

```rust
pub fn make_pq_dataset(
res: &Resources,
source: &impl CuvsDataset,
params: Option<&ProductQuantizerParams>,
) -> Result<PqDataset, CagraError>
```

Train an owning device PQ dataset (CAGRA-Q) from a device-padded source.

`params` may be `None` to use library defaults. Keep the returned dataset
alive while any index uses it, then attach with [`Index::update_dataset`].

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:38`_

## GraphBuildAlgo

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphBuildAlgo {
    /* variants omitted */
}
```

Algorithm for building the internal k-NN graph.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:50`_

## SearchAlgo

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchAlgo {
    /* variants omitted */
}
```

Search kernel implementation.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:90`_

## HashMode

```rust
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum HashMode {
    /* variants omitted */
}
```

Hash-table mode used during search.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:126`_

## CagraError

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CagraError {
    /* variants omitted */
}
```

Error type for CAGRA operations.

_Source: `rust/cuvs/src/neighbors/cagra/mod.rs:158`_
