---
slug: api-reference/rust-api-cuvs-ivf-flat-index-params
---

# Ivf Flat Index Params Module

_Rust module: `cuvs::ivf_flat::index_params`_

_Source: `rust/cuvs/src/ivf_flat/index_params.rs`_

## IndexParams

```rust
pub struct IndexParams(pub ffi::cuvsIvfFlatIndexParams_t); {
    /* private fields */
}
```

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/ivf_flat/index_params.rs:15` |
| `set_n_lists` | `rust/cuvs/src/ivf_flat/index_params.rs:24` |
| `set_metric` | `rust/cuvs/src/ivf_flat/index_params.rs:32` |
| `set_metric_arg` | `rust/cuvs/src/ivf_flat/index_params.rs:40` |
| `set_kmeans_n_iters` | `rust/cuvs/src/ivf_flat/index_params.rs:47` |
| `set_kmeans_trainset_fraction` | `rust/cuvs/src/ivf_flat/index_params.rs:57` |
| `set_add_data_on_build` | `rust/cuvs/src/ivf_flat/index_params.rs:68` |

### new

```rust
pub fn new() -> Result<IndexParams>
```

Returns a new IndexParams

_Source: `rust/cuvs/src/ivf_flat/index_params.rs:15`_

### set_n_lists

```rust
pub fn set_n_lists(self, n_lists: u32) -> IndexParams
```

The number of clusters used in the coarse quantizer.

_Source: `rust/cuvs/src/ivf_flat/index_params.rs:24`_

### set_metric

```rust
pub fn set_metric(self, metric: DistanceType) -> IndexParams
```

DistanceType to use for building the index

_Source: `rust/cuvs/src/ivf_flat/index_params.rs:32`_

### set_metric_arg

```rust
pub fn set_metric_arg(self, metric_arg: f32) -> IndexParams
```

The number of iterations searching for kmeans centers during index building.

_Source: `rust/cuvs/src/ivf_flat/index_params.rs:40`_

### set_kmeans_n_iters

```rust
pub fn set_kmeans_n_iters(self, kmeans_n_iters: u32) -> IndexParams
```

The number of iterations searching for kmeans centers during index building.

_Source: `rust/cuvs/src/ivf_flat/index_params.rs:47`_

### set_kmeans_trainset_fraction

```rust
pub fn set_kmeans_trainset_fraction(self, kmeans_trainset_fraction: f64) -> IndexParams
```

If kmeans_trainset_fraction is less than 1, then the dataset is
subsampled, and only n_samples * kmeans_trainset_fraction rows
are used for training.

_Source: `rust/cuvs/src/ivf_flat/index_params.rs:57`_

### set_add_data_on_build

```rust
pub fn set_add_data_on_build(self, add_data_on_build: bool) -> IndexParams
```

After training the coarse and fine quantizers, we will populate
the index with the dataset if add_data_on_build == true, otherwise
the index is left empty, and the extend method can be used
to add new vectors to the index.

_Source: `rust/cuvs/src/ivf_flat/index_params.rs:68`_

_Source: `rust/cuvs/src/ivf_flat/index_params.rs:11`_
