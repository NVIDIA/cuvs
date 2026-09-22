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

A CAGRA approximate nearest neighbor index borrowing caller-owned dataset storage.

**Methods**

| Name | Source |
| --- | --- |
| `build` | `rust/cuvs/src/neighbors/cagra/index.rs:70` |
| `build_from_dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:80` |
| `update_dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:105` |
| `search` | `rust/cuvs/src/neighbors/cagra/index.rs:134` |
| `search_filtered` | `rust/cuvs/src/neighbors/cagra/index.rs:154` |
| `serialize` | `rust/cuvs/src/neighbors/cagra/index.rs:195` |
| `serialize_to_hnswlib` | `rust/cuvs/src/neighbors/cagra/index.rs:215` |
| `deserialize_graph` | `rust/cuvs/src/neighbors/cagra/index.rs:220` |
| `deserialize_graph_and_dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:233` |

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:70`_

### build_from_dataset

```rust
pub fn build_from_dataset<'a, D>(
res: &Resources,
params: &IndexParams,
dataset: &'a D,
) -> Result<Index<'a>>
where
D: CuvsDataset + ?Sized,
```

Build from an owning dataset or non-owning dataset view.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:80`_

### update_dataset

```rust
pub fn update_dataset<'a, D>(self, res: &Resources, dataset: &'a D) -> Result<Index<'a>>
where
D: CuvsDataset + ?Sized,
```

Attach a device-padded dataset and return a search-ready index borrowing it.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:105`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:134`_

### search_filtered

```rust
pub fn search_filtered<Q, N, D>(
&self,
res: &Resources,
params: &SearchParams,
queries: &Q,
neighbors: &mut N,
distances: &mut D,
filter: &Filter<'_, Bitset>,
) -> Result<()>
where
Q: AsDlTensor + ?Sized,
N: AsDlTensorMut + ?Sized,
D: AsDlTensorMut + ?Sized,
```

Searches the index with a row-level bitset filter.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:154`_

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

Deserialize a graph-only file with [`Index::deserialize_graph`], or
recreate the serialized dataset's residency and layout with
[`Index::deserialize_graph_and_dataset`].

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:195`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:215`_

### deserialize_graph

```rust
pub fn deserialize_graph<P: AsRef<Path>>(
res: &Resources,
filename: P,
) -> Result<DeserializedIndex<Dataset>>
```

Load only the graph, ignoring any dataset stored in the file.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:220`_

### deserialize_graph_and_dataset

```rust
pub fn deserialize_graph_and_dataset<P: AsRef<Path>>(
res: &Resources,
filename: P,
) -> Result<DeserializedIndex<Dataset>>
```

Load the graph and recreate its serialized dataset allocation.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:233`_

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:47`_

## DeserializedIndex

```rust
#[derive(Debug)]
pub struct DeserializedIndex<D> {
    /* private fields */
}
```

A deserialized CAGRA index and the optional dataset storage it views.

A file serialized without vectors yields `dataset == None` and must have
matching storage attached before search. Field order is significant: the
native index is destroyed before its dataset owner.

**Methods**

| Name | Source |
| --- | --- |
| `dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:254` |
| `has_dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:259` |
| `serialize` | `rust/cuvs/src/neighbors/cagra/index.rs:264` |
| `serialize_to_hnswlib` | `rust/cuvs/src/neighbors/cagra/index.rs:274` |
| `update_dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:279` |
| `search` | `rust/cuvs/src/neighbors/cagra/index.rs:304` |
| `search_filtered` | `rust/cuvs/src/neighbors/cagra/index.rs:325` |

### dataset

```rust
pub fn dataset(&self) -> Option<&D>
```

Borrow the dataset owner when the serialized file included vectors.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:254`_

### has_dataset

```rust
pub fn has_dataset(&self) -> bool
```

Whether the serialized file included vector storage.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:259`_

### serialize

```rust
pub fn serialize<P: AsRef<Path>>(
&self,
res: &Resources,
filename: P,
include_dataset: bool,
) -> Result<()>
```

Save this index to file.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:264`_

### serialize_to_hnswlib

```rust
pub fn serialize_to_hnswlib<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()>
```

Save this index to file in the cuVS hnswlib format.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:274`_

### update_dataset

```rust
pub fn update_dataset<'a, T>(self, res: &Resources, dataset: &'a T) -> Result<Index<'a>>
where
T: CuvsDataset + ?Sized,
```

Replace the deserialized storage with a caller-owned device-padded view.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:279`_

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

Search an index whose deserialized owner is device-padded.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:304`_

### search_filtered

```rust
pub fn search_filtered<Q, N, D>(
&self,
res: &Resources,
params: &SearchParams,
queries: &Q,
neighbors: &mut N,
distances: &mut D,
filter: &Filter<'_, Bitset>,
) -> Result<()>
where
Q: AsDlTensor + ?Sized,
N: AsDlTensorMut + ?Sized,
D: AsDlTensorMut + ?Sized,
```

Search a padded deserialized index with a row-level bitset filter.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:325`_

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:58`_
