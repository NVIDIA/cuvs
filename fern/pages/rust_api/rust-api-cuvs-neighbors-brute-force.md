---
slug: api-reference/rust-api-cuvs-neighbors-brute-force
---

# Neighbors Brute Force Module

_Rust module: `cuvs::neighbors::brute_force`_

_Source: `rust/cuvs/src/neighbors/brute_force.rs`_

Brute-force (exact) k-NN.

Build an [`Index`] over a dataset, then [`search`](Index::search) it with
device-resident queries and output buffers. Tensors are borrowed through the
`AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
module for the tensor model and `examples/cagra.rs` for the same
build/search workflow.

## crate::neighbors::filters::SearchFilter

```rust
pub use crate::neighbors::filters::SearchFilter;
```

_Source: `rust/cuvs/src/neighbors/brute_force.rs:19`_

## BruteForceError

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BruteForceError {
    /* variants omitted */
}
```

Error type for brute-force operations.

_Source: `rust/cuvs/src/neighbors/brute_force.rs:28`_

## Index

```rust
#[derive(Debug)]
pub struct Index<'d> {
    /* private fields */
}
```

Brute-force KNN index.

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/neighbors/brute_force.rs:54` |
| `search` | `rust/cuvs/src/neighbors/brute_force.rs:86` |
| `search_filtered` | `rust/cuvs/src/neighbors/brute_force.rs:105` |

### build

```rust
pub fn build<T>(res: &Resources, metric: DistanceType, dataset: &'d T) -> Result<Index<'d>>
where
T: AsDlTensor + ?Sized,
```

Builds a brute-force index over `dataset` for exact k-NN search.

`metric` selects the distance (use [`DistanceType::LpUnexpanded`] to set
the Minkowski exponent `p`). `dataset` is a row-major matrix on the host
or device implementing [`AsDlTensor`]; the C++ index keeps a non-owning
view of it, so the returned [`Index`] borrows it for `'d` and cannot
outlive it.

_Source: `rust/cuvs/src/neighbors/brute_force.rs:54`_

### search

```rust
pub fn search<Q, N, D>(
&self,
res: &Resources,
queries: &Q,
neighbors: &mut N,
distances: &mut D,
) -> Result<()>
where
Q: AsDlTensor + ?Sized,
N: AsDlTensorMut + ?Sized,
D: AsDlTensorMut + ?Sized,
```

Searches the index for the `k` nearest neighbors of each query.

`queries`, `neighbors`, and `distances` must reside in device memory and
implement [`AsDlTensor`] / [`AsDlTensorMut`]. `neighbors` receives the
neighbor indices and `distances` their distances; both are written in
place.

_Source: `rust/cuvs/src/neighbors/brute_force.rs:86`_

### search_filtered

```rust
pub fn search_filtered<Q, N, D>(
&self,
res: &Resources,
queries: &Q,
neighbors: &mut N,
distances: &mut D,
filter: &SearchFilter<'_>,
) -> Result<()>
where
Q: AsDlTensor + ?Sized,
N: AsDlTensorMut + ?Sized,
D: AsDlTensorMut + ?Sized,
```

Searches the index using a row bitset or per-query bitmap filter.

_Source: `rust/cuvs/src/neighbors/brute_force.rs:105`_

_Source: `rust/cuvs/src/neighbors/brute_force.rs:39`_
