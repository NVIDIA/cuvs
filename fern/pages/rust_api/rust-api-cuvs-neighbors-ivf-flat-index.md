---
slug: api-reference/rust-api-cuvs-neighbors-ivf-flat-index
---

# Neighbors Ivf Flat Index Module

_Rust module: `cuvs::neighbors::ivf_flat::index`_

_Source: `rust/cuvs/src/neighbors/ivf_flat/index.rs`_

## Index

```rust
#[derive(Debug)]
pub struct Index(ffi::cuvsIvfFlatIndex_t); {
    /* private fields */
}
```

IVF-Flat ANN index.

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/neighbors/ivf_flat/index.rs:29` |
| `search` | `rust/cuvs/src/neighbors/ivf_flat/index.rs:60` |
| `search_filtered` | `rust/cuvs/src/neighbors/ivf_flat/index.rs:80` |

### build

```rust
pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
where
T: AsDlTensor + ?Sized,
```

Builds an IVF-Flat index over `dataset` for efficient search.

`dataset` is a row-major matrix on the host or device implementing
[`AsDlTensor`]. It is copied into the index, so the caller may free it
once this call returns (hence `Index` carries no lifetime).

Supported dataset/query dtypes in the current C-backed implementation are
`f32`, `f16`, `i8`, and `u8`.

_Source: `rust/cuvs/src/neighbors/ivf_flat/index.rs:29`_

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

_Source: `rust/cuvs/src/neighbors/ivf_flat/index.rs:60`_

### search_filtered

```rust
pub fn search_filtered<Q, N, D>(
&self,
res: &Resources,
params: &SearchParams,
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

Searches the index with a row-level bitset filter.

_Source: `rust/cuvs/src/neighbors/ivf_flat/index.rs:80`_

_Source: `rust/cuvs/src/neighbors/ivf_flat/index.rs:18`_
