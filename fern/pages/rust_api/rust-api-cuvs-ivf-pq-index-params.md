---
slug: api-reference/rust-api-cuvs-ivf-pq-index-params
---

# Ivf Pq Index Params Module

_Rust module: `cuvs::ivf_pq::index_params`_

_Source: `rust/cuvs/src/ivf_pq/index_params.rs`_

## ffi::cuvsIvfPqCodebookGen

```rust
pub use ffi::cuvsIvfPqCodebookGen;
```

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:11`_

## ffi::cuvsIvfPqListLayout

```rust
pub use ffi::cuvsIvfPqListLayout;
```

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:12`_

## IndexParams

```rust
pub struct IndexParams(pub ffi::cuvsIvfPqIndexParams_t); {
    /* private fields */
}
```

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/ivf_pq/index_params.rs:18` |
| `set_n_lists` | `rust/cuvs/src/ivf_pq/index_params.rs:27` |
| `set_metric` | `rust/cuvs/src/ivf_pq/index_params.rs:35` |
| `set_metric_arg` | `rust/cuvs/src/ivf_pq/index_params.rs:43` |
| `set_kmeans_n_iters` | `rust/cuvs/src/ivf_pq/index_params.rs:51` |
| `set_kmeans_trainset_fraction` | `rust/cuvs/src/ivf_pq/index_params.rs:61` |
| `set_pq_bits` | `rust/cuvs/src/ivf_pq/index_params.rs:69` |
| `set_pq_dim` | `rust/cuvs/src/ivf_pq/index_params.rs:85` |
| `set_codebook_kind` | `rust/cuvs/src/ivf_pq/index_params.rs:92` |
| `set_codes_layout` | `rust/cuvs/src/ivf_pq/index_params.rs:103` |
| `set_force_random_rotation` | `rust/cuvs/src/ivf_pq/index_params.rs:121` |
| `set_max_train_points_per_pq_code` | `rust/cuvs/src/ivf_pq/index_params.rs:133` |
| `set_add_data_on_build` | `rust/cuvs/src/ivf_pq/index_params.rs:144` |

### new

```rust
pub fn new() -> Result<IndexParams>
```

Returns a new IndexParams

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:18`_

### set_n_lists

```rust
pub fn set_n_lists(self, n_lists: u32) -> IndexParams
```

The number of clusters used in the coarse quantizer.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:27`_

### set_metric

```rust
pub fn set_metric(self, metric: DistanceType) -> IndexParams
```

DistanceType to use for building the index

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:35`_

### set_metric_arg

```rust
pub fn set_metric_arg(self, metric_arg: f32) -> IndexParams
```

The number of iterations searching for kmeans centers during index building.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:43`_

### set_kmeans_n_iters

```rust
pub fn set_kmeans_n_iters(self, kmeans_n_iters: u32) -> IndexParams
```

The number of iterations searching for kmeans centers during index building.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:51`_

### set_kmeans_trainset_fraction

```rust
pub fn set_kmeans_trainset_fraction(self, kmeans_trainset_fraction: f64) -> IndexParams
```

If kmeans_trainset_fraction is less than 1, then the dataset is
subsampled, and only n_samples * kmeans_trainset_fraction rows
are used for training.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:61`_

### set_pq_bits

```rust
pub fn set_pq_bits(self, pq_bits: u32) -> IndexParams
```

The bit length of the vector element after quantization.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:69`_

### set_pq_dim

```rust
pub fn set_pq_dim(self, pq_dim: u32) -> IndexParams
```

The dimensionality of a the vector after product quantization.
When zero, an optimal value is selected using a heuristic. Note
pq_dim * pq_bits must be a multiple of 8. Hint: a smaller 'pq_dim'
results in a smaller index size and better search performance, but
lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number,
but multiple of 8 are desirable for good performance. If 'pq_bits'
is not 8, 'pq_dim' should be a multiple of 8. For good performance,
it is desirable that 'pq_dim' is a multiple of 32. Ideally,
'pq_dim' should be also a divisor of the dataset dim.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:85`_

### set_codebook_kind

```rust
pub fn set_codebook_kind(self, codebook_kind: cuvsIvfPqCodebookGen) -> IndexParams
```

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:92`_

### set_codes_layout

```rust
pub fn set_codes_layout(self, codes_layout: cuvsIvfPqListLayout) -> IndexParams
```

Memory layout of the IVF-PQ list data.
- FLAT: Codes are stored contiguously, one vector's codes after another.
- INTERLEAVED: Codes are interleaved for optimized search performance.
This is the default and recommended for search workloads.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:103`_

### set_force_random_rotation

```rust
pub fn set_force_random_rotation(self, force_random_rotation: bool) -> IndexParams
```

Apply a random rotation matrix on the input data and queries even
if `dim % pq_dim == 0`. Note: if `dim` is not multiple of `pq_dim`,
a random rotation is always applied to the input data and queries
to transform the working space from `dim` to `rot_dim`, which may
be slightly larger than the original space and and is a multiple
of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is
not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`,
hence no need in adding "extra" data columns / features). By
default, if `dim == rot_dim`, the rotation transform is
initialized with the identity matrix. When
`force_random_rotation == True`, a random orthogonal transform

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:121`_

### set_max_train_points_per_pq_code

```rust
pub fn set_max_train_points_per_pq_code(self, max_pq_points: u32) -> IndexParams
```

The max number of data points to use per PQ code during PQ codebook training. Using more data
points per PQ code may increase the quality of PQ codebook but may also increase the build
time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and
PER_CLUSTER. In both cases, we will use `pq_book_size * max_train_points_per_pq_code` training
points to train each codebook.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:133`_

### set_add_data_on_build

```rust
pub fn set_add_data_on_build(self, add_data_on_build: bool) -> IndexParams
```

After training the coarse and fine quantizers, we will populate
the index with the dataset if add_data_on_build == true, otherwise
the index is left empty, and the extend method can be used
to add new vectors to the index.

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:144`_

_Source: `rust/cuvs/src/ivf_pq/index_params.rs:14`_
