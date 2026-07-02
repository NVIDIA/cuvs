---
slug: api-reference/rust-api-cuvs-brute-force
---

# Brute Force Module

_Rust module: `cuvs::brute_force`_

_Source: `rust/cuvs/src/brute_force.rs`_

Brute Force KNN

## Index

```rust
#[derive(Debug)]
pub struct Index {
    /* private fields */
}
```

Brute Force KNN Index

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/brute_force.rs:32` |
| `new` | `rust/cuvs/src/brute_force.rs:54` |
| `search` | `rust/cuvs/src/brute_force.rs:70` |

### build

```rust
pub fn build<T: Into<ManagedTensor>>(
res: &Resources,
metric: DistanceType,
metric_arg: Option<f32>,
dataset: T,
) -> Result<Index>
```

Builds a new Brute Force KNN Index from the dataset for efficient search.

#### Arguments

* `res` - Resources to use
* `metric` - DistanceType to use for building the index
* `metric_arg` - Optional value of `p` for Minkowski distances
* `dataset` - A row-major matrix on either the host or device to index

_Source: `rust/cuvs/src/brute_force.rs:32`_

### new

```rust
pub fn new() -> Result<Index>
```

Creates a new empty index

_Source: `rust/cuvs/src/brute_force.rs:54`_

### search

```rust
pub fn search(
&self,
res: &Resources,
queries: &ManagedTensor,
neighbors: &ManagedTensor,
distances: &ManagedTensor,
) -> Result<()>
```

Perform a Nearest Neighbors search on the Index

#### Arguments

* `res` - Resources to use
* `queries` - A matrix in device memory to query for
* `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
* `distances` - Matrix in device memory that receives the distances of the nearest neighbors

_Source: `rust/cuvs/src/brute_force.rs:70`_

_Source: `rust/cuvs/src/brute_force.rs:16`_
