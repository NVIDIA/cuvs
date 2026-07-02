---
slug: api-reference/rust-api-cuvs-neighbors-cagra-index
---

# Neighbors Cagra Index Module

_Rust module: `cuvs::neighbors::cagra::index`_

_Source: `rust/cuvs/src/neighbors/cagra/index.rs`_

## Index

```rust
#[derive(Debug)]
pub struct Index<'d> {
    /* private fields */
}
```

A CAGRA approximate nearest neighbor index.

The lifetime `'d` ties this index to the underlying dataset,
passed at construction time. The C library may store a non-owning view
of properly aligned device-resident data, so the dataset must outlive
the index. When an index is deserialized from disk, the data is
self-contained and its lifetime is `'static`.

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/neighbors/cagra/index.rs:44` |
| `new` | `rust/cuvs/src/neighbors/cagra/index.rs:62` |
| `search` | `rust/cuvs/src/neighbors/cagra/index.rs:77` |
| `search_with_filter` | `rust/cuvs/src/neighbors/cagra/index.rs:117` |
| `serialize` | `rust/cuvs/src/neighbors/cagra/index.rs:190` |
| `serialize_to_hnswlib` | `rust/cuvs/src/neighbors/cagra/index.rs:214` |
| `deserialize` | `rust/cuvs/src/neighbors/cagra/index.rs:230` |

### build

```rust
pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &'d T) -> Result<Index<'d>>
where
T: AsDlTensor + ?Sized,
```

Builds a CAGRA index over `dataset` for efficient search.

`dataset` is a row-major matrix on the host or device implementing
[`AsDlTensor`]. The C++ index keeps a non-owning
view of it, so the returned [`Index`] borrows `dataset` for `'d` and
cannot outlive it.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:44`_

### new

```rust
pub fn new() -> Result<Index<'d>>
```

Creates a new empty index

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:62`_

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
implement [`AsDlTensor`] /
[`AsDlTensorMut`]. `neighbors` (shape
`n_queries × k`) receives the neighbor indices and `distances` their
distances; both are written in place.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:77`_

### search_with_filter

```rust
pub fn search_with_filter<Q, N, D, B>(
&self,
res: &Resources,
params: &SearchParams,
queries: &Q,
neighbors: &mut N,
distances: &mut D,
bitset: &B,
) -> Result<()>
where
Q: AsDlTensor + ?Sized,
N: AsDlTensorMut + ?Sized,
D: AsDlTensorMut + ?Sized,
B: AsDlTensor + ?Sized,
```

Perform a filtered Approximate Nearest Neighbors search on the Index

Like [`search`](Self::search), but applies a bitset filter to exclude
vectors during graph traversal. Filtered vectors are never visited, which
gives better recall than post-filtering.

`queries`, `neighbors`, and `distances` are as in [`search`](Self::search).
`bitset` is a 1-D `uint32` device tensor of `ceil(n_rows / 32)` elements,
where each bit maps to a dataset row (1 = include, 0 = exclude).

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:117`_

### serialize

```rust
pub fn serialize<P: AsRef<Path>>(
&self,
res: &Resources,
filename: P,
include_dataset: bool,
) -> Result<()>
```

Save the CAGRA index to file.

Experimental, both the API and the serialization format are subject to change.

#### Arguments

* `res` - Resources to use
* `filename` - The file path for saving the index
* `include_dataset` - Whether to write out the dataset to the file

#### Example:
```no_run
use cuvs::Resources;
use cuvs::neighbors::cagra::{Index, IndexParams};

fn serialize_example() -> Result<(), Box<dyn std::error::Error>> {
let res = Resources::new()?;

// Build an index (using some dataset)
let build_params = IndexParams::builder().build()?;
// let index = Index::build(&res, &build_params, &dataset)?;

// Save the index to disk (including the dataset)
// index.serialize(&res, "/path/to/index.bin", true)?;

// Later, load the index from disk
let loaded_index = Index::deserialize(&res, "/path/to/index.bin")?;

// The loaded index can be used for search just like the original
Ok(())
}
```

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:190`_

### serialize_to_hnswlib

```rust
pub fn serialize_to_hnswlib<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()>
```

Save the CAGRA index to file in hnswlib format.

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

#### Arguments

* `res` - Resources to use
* `filename` - The file path for saving the index

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:214`_

### deserialize

```rust
pub fn deserialize<P: AsRef<Path>>(res: &Resources, filename: P) -> Result<Index<'static>>
```

Load a CAGRA index from file.

Experimental, both the API and the serialization format are subject to change.

#### Arguments

* `res` - Resources to use
* `filename` - The path of the file that stores the index

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:230`_

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:26`_
