---
slug: api-reference/rust-api-cuvs-ivf-flat-index
---

# Ivf Flat Index Module

_Rust module: `cuvs::ivf_flat::index`_

_Source: `rust/cuvs/src/ivf_flat/index.rs`_

## Index

```rust
#[derive(Debug)]
pub struct Index(ffi::cuvsIvfFlatIndex_t); {
    /* private fields */
}
```

Ivf-Flat ANN Index

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/ivf_flat/index.rs:25` |
| `new` | `rust/cuvs/src/ivf_flat/index.rs:39` |
| `search` | `rust/cuvs/src/ivf_flat/index.rs:56` |

### build

```rust
pub fn build<T: Into<ManagedTensor>>(
res: &Resources,
params: &IndexParams,
dataset: T,
) -> Result<Index>
```

Builds a new Index from the dataset for efficient search.

#### Arguments

* `res` - Resources to use
* `params` - Parameters for building the index
* `dataset` - A row-major matrix on either the host or device to index

_Source: `rust/cuvs/src/ivf_flat/index.rs:25`_

### new

```rust
pub fn new() -> Result<Index>
```

Creates a new empty index

_Source: `rust/cuvs/src/ivf_flat/index.rs:39`_

### search

```rust
pub fn search(
&self,
res: &Resources,
params: &SearchParams,
queries: &ManagedTensor,
neighbors: &ManagedTensor,
distances: &ManagedTensor,
) -> Result<()>
```

Perform a Approximate Nearest Neighbors search on the Index

#### Arguments

* `res` - Resources to use
* `params` - Parameters to use in searching the index
* `queries` - A matrix in device memory to query for
* `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
* `distances` - Matrix in device memory that receives the distances of the nearest neighbors

_Source: `rust/cuvs/src/ivf_flat/index.rs:56`_

_Source: `rust/cuvs/src/ivf_flat/index.rs:15`_
