---
slug: api-reference/rust-api-cuvs-vamana-index
---

# Vamana Index Module

_Rust module: `cuvs::vamana::index`_

_Source: `rust/cuvs/src/vamana/index.rs`_

## Index

```rust
#[derive(Debug)]
pub struct Index(ffi::cuvsVamanaIndex_t); {
    /* private fields */
}
```

Vamana ANN Index

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/vamana/index.rs:33` |
| `new` | `rust/cuvs/src/vamana/index.rs:47` |
| `serialize` | `rust/cuvs/src/vamana/index.rs:66` |

### build

```rust
pub fn build<T: Into<ManagedTensor>>(
res: &Resources,
params: &IndexParams,
dataset: T,
) -> Result<Index>
```

Builds Vamana Index for efficient DiskANN search

The build uses the Vamana insertion-based algorithm to create the graph. The algorithm
starts with an empty graph and iteratively inserts batches of nodes. Each batch involves
performing a greedy search for each vector to be inserted, and inserting it with edges to
all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied
to improve graph quality. The index_params struct controls the degree of the final graph.


#### Arguments

* `res` - Resources to use
* `params` - Parameters for building the index
* `dataset` - A row-major matrix on either the host or device to index

_Source: `rust/cuvs/src/vamana/index.rs:33`_

### new

```rust
pub fn new() -> Result<Index>
```

Creates a new empty index

_Source: `rust/cuvs/src/vamana/index.rs:47`_

### serialize

```rust
pub fn serialize(self, res: &Resources, filename: &str, include_dataset: bool) -> Result<()>
```

Save Vamana index to file

Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility.

Serialized Index is to be used by the DiskANN open-source repository for graph search.

#### Arguments

* `res` - Resources to use
* `filename` - The file prefix for where the index is sazved
* `include_dataset` - whether to include the dataset in the serialized index

_Source: `rust/cuvs/src/vamana/index.rs:66`_

_Source: `rust/cuvs/src/vamana/index.rs:16`_
