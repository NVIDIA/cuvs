---
slug: api-reference/rust-api-cuvs-neighbors-ivf-pq-index
---

# Neighbors Ivf Pq Index Module

_Rust module: `cuvs::neighbors::ivf_pq::index`_

_Source: `rust/cuvs/src/neighbors/ivf_pq/index.rs`_

## Index

```rust
#[derive(Debug)]
pub struct Index(ffi::cuvsIvfPqIndex_t); {
    /* private fields */
}
```

IVF-PQ ANN index.

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/neighbors/ivf_pq/index.rs:25` |
| `new` | `rust/cuvs/src/neighbors/ivf_pq/index.rs:43` |
| `search` | `rust/cuvs/src/neighbors/ivf_pq/index.rs:57` |

### build

```rust
pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
where
T: AsDlTensor + ?Sized,
```

Builds an IVF-PQ index over `dataset` for compressed, efficient search.

`dataset` is a row-major matrix on the host or device implementing
[`AsDlTensor`]. It is copied into the index, so the caller may free it
once this call returns (hence `Index` carries no lifetime).

_Source: `rust/cuvs/src/neighbors/ivf_pq/index.rs:25`_

### new

```rust
pub fn new() -> Result<Index>
```

Creates a new empty index.

_Source: `rust/cuvs/src/neighbors/ivf_pq/index.rs:43`_

### search

```rust
pub fn search<Q, N, D>(
&self,
res: &Resources,
params: &SearchParams,
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

_Source: `rust/cuvs/src/neighbors/ivf_pq/index.rs:57`_

_Source: `rust/cuvs/src/neighbors/ivf_pq/index.rs:17`_
