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
| `merge` | `rust/cuvs/src/neighbors/cagra/index.rs:138` |
| `merge_filtered` | `rust/cuvs/src/neighbors/cagra/index.rs:156` |
| `merge_with_params` | `rust/cuvs/src/neighbors/cagra/index.rs:173` |
| `merge_filtered_with_params` | `rust/cuvs/src/neighbors/cagra/index.rs:193` |
| `search` | `rust/cuvs/src/neighbors/cagra/index.rs:280` |
| `search_filtered` | `rust/cuvs/src/neighbors/cagra/index.rs:300` |
| `serialize` | `rust/cuvs/src/neighbors/cagra/index.rs:341` |
| `serialize_to_hnswlib` | `rust/cuvs/src/neighbors/cagra/index.rs:361` |
| `deserialize_graph` | `rust/cuvs/src/neighbors/cagra/index.rs:366` |
| `deserialize_graph_and_dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:379` |

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

### merge

```rust
pub fn merge<'a, D>(
res: &Resources,
params: &IndexParams,
indices: &[&Index<'_>],
merged_dataset: &'a D,
offsets: &[i64],
) -> Result<Index<'a>>
where
D: CuvsDataset + ?Sized,
```

Merges multiple CAGRA indices into a new index backed by `merged_dataset`.

The caller must have already concatenated every input index's dataset (in
`indices` order) into `merged_dataset`, and computed `offsets` as the
cumulative row counts of each input index: `offsets[i]` is the row at
which `indices[i]`'s rows start in `merged_dataset`, and
`offsets[indices.len()]` must equal `merged_dataset`'s total row count.
See [`merged_dataset_offsets`] for the bitset-filtered case.

The returned [`Index`] borrows `merged_dataset` for `'a` and cannot
outlive it, mirroring [`Index::update_dataset`].

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:138`_

### merge_filtered

```rust
pub fn merge_filtered<'a, D>(
res: &Resources,
params: &IndexParams,
indices: &[&Index<'_>],
filter: &Filter<'_, Bitset>,
merged_dataset: &'a D,
offsets: &[i64],
) -> Result<Index<'a>>
where
D: CuvsDataset + ?Sized,
```

Merges multiple CAGRA indices, applying a row-level bitset `filter`.

`merged_dataset` must already contain only the rows surviving `filter`
(in `indices` order); use [`merged_dataset_offsets`] to compute the
per-index row offsets within it.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:156`_

### merge_with_params

```rust
pub fn merge_with_params<'a, D>(
res: &Resources,
params: &IndexParams,
merge_params: &MergeParams,
indices: &[&Index<'_>],
merged_dataset: &'a D,
offsets: &[i64],
) -> Result<Index<'a>>
where
D: CuvsDataset + ?Sized,
```

Merges multiple CAGRA indices using explicit [`MergeParams`].

See [`Index::merge`] for the `merged_dataset`/`offsets` contract.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:173`_

### merge_filtered_with_params

```rust
#[allow(clippy::too_many_arguments)]
pub fn merge_filtered_with_params<'a, D>(
res: &Resources,
params: &IndexParams,
merge_params: &MergeParams,
indices: &[&Index<'_>],
filter: &Filter<'_, Bitset>,
merged_dataset: &'a D,
offsets: &[i64],
) -> Result<Index<'a>>
where
D: CuvsDataset + ?Sized,
```

Merges multiple CAGRA indices using explicit [`MergeParams`] and a
row-level bitset `filter`.

See [`Index::merge`] and [`Index::merge_filtered`] for the
`merged_dataset`/`offsets` contract.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:193`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:280`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:300`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:341`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:361`_

### deserialize_graph

```rust
pub fn deserialize_graph<P: AsRef<Path>>(
res: &Resources,
filename: P,
) -> Result<DeserializedIndex<Dataset>>
```

Load only the graph, ignoring any dataset stored in the file.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:366`_

### deserialize_graph_and_dataset

```rust
pub fn deserialize_graph_and_dataset<P: AsRef<Path>>(
res: &Resources,
filename: P,
) -> Result<DeserializedIndex<Dataset>>
```

Load the graph and recreate its serialized dataset allocation.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:379`_

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
| `dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:400` |
| `has_dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:405` |
| `serialize` | `rust/cuvs/src/neighbors/cagra/index.rs:410` |
| `serialize_to_hnswlib` | `rust/cuvs/src/neighbors/cagra/index.rs:420` |
| `update_dataset` | `rust/cuvs/src/neighbors/cagra/index.rs:425` |
| `search` | `rust/cuvs/src/neighbors/cagra/index.rs:450` |
| `search_filtered` | `rust/cuvs/src/neighbors/cagra/index.rs:471` |

### dataset

```rust
pub fn dataset(&self) -> Option<&D>
```

Borrow the dataset owner when the serialized file included vectors.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:400`_

### has_dataset

```rust
pub fn has_dataset(&self) -> bool
```

Whether the serialized file included vector storage.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:405`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:410`_

### serialize_to_hnswlib

```rust
pub fn serialize_to_hnswlib<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()>
```

Save this index to file in the cuVS hnswlib format.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:420`_

### update_dataset

```rust
pub fn update_dataset<'a, T>(self, res: &Resources, dataset: &'a T) -> Result<Index<'a>>
where
T: CuvsDataset + ?Sized,
```

Replace the deserialized storage with a caller-owned device-padded view.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:425`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:450`_

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

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:471`_

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:58`_

## merged_dataset_offsets

```rust
pub fn merged_dataset_offsets(
res: &Resources,
indices: &[&Index<'_>],
filter: Option<&Filter<'_, Bitset>>,
) -> Result<Vec<i64>>
```

Computes per-index row offsets within a to-be-built merge buffer.

`cuvsCagraMerge`/`Index::merge` require the caller to have already
concatenated every input index's dataset (in `indices` order, applying
`filter` if any) into a single buffer, and to know each index's starting
row within it. For an unfiltered merge those offsets are just the
cumulative row counts of `indices`, so this function is unnecessary. For a
bitset `filter`, the number of surviving rows per index cannot be derived
any other way, so call this first.

Returns a `Vec` of `indices.len() + 1` entries: entry `i` is the row at
which `indices[i]`'s surviving rows must start in the merged buffer; the
last entry is the total row count of the merged buffer.

_Source: `rust/cuvs/src/neighbors/cagra/index.rs:528`_
