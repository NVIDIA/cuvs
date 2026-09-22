---
slug: api-reference/rust-api-cuvs-neighbors-vamana-index
---

# Neighbors Vamana Index Module

_Rust module: `cuvs::neighbors::vamana::index`_

_Source: `rust/cuvs/src/neighbors/vamana/index.rs`_

## Index

```rust
#[derive(Debug)]
pub struct Index(ffi::cuvsVamanaIndex_t); {
    /* private fields */
}
```

Vamana ANN index.

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/neighbors/vamana/index.rs:33` |
| `serialize` | `rust/cuvs/src/neighbors/vamana/index.rs:65` |

### build

```rust
pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
where
T: AsDlTensor + ?Sized,
```

Builds a Vamana index for efficient DiskANN search.

The build uses the Vamana insertion-based algorithm: starting from an
empty graph it iteratively inserts batches of nodes, performing a greedy
search for each inserted vector and connecting it to all nodes traversed;
reverse edges are added and `robustPrune` is applied to improve quality.
[`IndexParams`] controls the degree of the final graph.

`dataset` is a row-major matrix on the host or device implementing
[`AsDlTensor`]; it is copied into the index, so `Index` carries no
lifetime.

_Source: `rust/cuvs/src/neighbors/vamana/index.rs:33`_

### serialize

```rust
pub fn serialize(
&self,
res: &Resources,
filename: impl AsRef<Path>,
include_dataset: bool,
) -> Result<()>
```

Saves the Vamana index to a file.

Matches the on-disk format used by the DiskANN open-source repository,
so the serialized index can be consumed there for graph search.

`filename` is the file prefix under which the index is saved;
`include_dataset` controls whether the dataset is embedded.

_Source: `rust/cuvs/src/neighbors/vamana/index.rs:65`_

_Source: `rust/cuvs/src/neighbors/vamana/index.rs:19`_
