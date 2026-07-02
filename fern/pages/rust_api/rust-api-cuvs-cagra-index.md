---
slug: api-reference/rust-api-cuvs-cagra-index
---

# Cagra Index Module

_Rust module: `cuvs::cagra::index`_

_Source: `rust/cuvs/src/cagra/index.rs`_

## Index

```rust
#[derive(Debug)]
pub struct Index(ffi::cuvsCagraIndex_t); {
    /* private fields */
}
```

CAGRA ANN Index

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/cagra/index.rs:38` |
| `new` | `rust/cuvs/src/cagra/index.rs:52` |
| `search` | `rust/cuvs/src/cagra/index.rs:69` |
| `search_with_filter` | `rust/cuvs/src/cagra/index.rs:107` |
| `serialize` | `rust/cuvs/src/cagra/index.rs:143` |
| `serialize_to_hnswlib` | `rust/cuvs/src/cagra/index.rs:166` |
| `deserialize` | `rust/cuvs/src/cagra/index.rs:179` |

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

_Source: `rust/cuvs/src/cagra/index.rs:38`_

### new

```rust
pub fn new() -> Result<Index>
```

Creates a new empty index

_Source: `rust/cuvs/src/cagra/index.rs:52`_

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

_Source: `rust/cuvs/src/cagra/index.rs:69`_

### search_with_filter

```rust
pub fn search_with_filter(
&self,
res: &Resources,
params: &SearchParams,
queries: &ManagedTensor,
neighbors: &ManagedTensor,
distances: &ManagedTensor,
bitset: &ManagedTensor,
) -> Result<()>
```

Perform a filtered Approximate Nearest Neighbors search on the Index

Like [`search`](Self::search), but accepts a bitset filter to exclude
vectors during graph traversal. Filtered vectors are never visited,
giving better recall than post-filtering.

#### Arguments

* `res` - Resources to use
* `params` - Parameters to use in searching the index
* `queries` - A matrix in device memory to query for
* `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
* `distances` - Matrix in device memory that receives the distances of the nearest neighbors
* `bitset` - A 1-D `uint32` device tensor with `ceil(n_rows / 32)` elements.
Each bit corresponds to a dataset row: bit 1 = include, bit 0 = exclude.

_Source: `rust/cuvs/src/cagra/index.rs:107`_

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

_Source: `rust/cuvs/src/cagra/index.rs:143`_

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

_Source: `rust/cuvs/src/cagra/index.rs:166`_

### deserialize

```rust
pub fn deserialize<P: AsRef<Path>>(res: &Resources, filename: P) -> Result<Index>
```

Load a CAGRA index from file.

Experimental, both the API and the serialization format are subject to change.

#### Arguments

* `res` - Resources to use
* `filename` - The path of the file that stores the index

_Source: `rust/cuvs/src/cagra/index.rs:179`_

_Source: `rust/cuvs/src/cagra/index.rs:17`_
