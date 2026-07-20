---
slug: api-reference/rust-api-cuvs-neighbors-ivf-pq-params
---

# Neighbors Ivf Pq Params Module

_Rust module: `cuvs::neighbors::ivf_pq::params`_

_Source: `rust/cuvs/src/neighbors/ivf_pq/params.rs`_

Builder-pattern parameter types for IVF-PQ build and search.

All setters are optional; unset values retain the library defaults from the
underlying C `*ParamsCreate` functions.

## IndexParams

```rust
pub struct IndexParams {
    /* private fields */
}
```

Parameters for building an IVF-PQ index.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/ivf_pq/params.rs:29` |
| `try_new` | `rust/cuvs/src/neighbors/ivf_pq/params.rs:112` |

### new

```rust
#[builder]
#[allow(clippy::too_many_arguments)]
pub fn new(
n_lists: Option<u32>,
metric: Option<DistanceType>,
kmeans_n_iters: Option<u32>,
kmeans_trainset_fraction: Option<f64>,
pq_bits: Option<u32>,
pq_dim: Option<u32>,
codebook_kind: Option<CodebookGen>,
codes_layout: Option<ListLayout>,
force_random_rotation: Option<bool>,
max_train_points_per_pq_code: Option<u32>,
add_data_on_build: Option<bool>,
) -> Result<Self, IvfPqError>
```

_Source: `rust/cuvs/src/neighbors/ivf_pq/params.rs:29`_

### try_new

```rust
pub fn try_new() -> Result<Self, IvfPqError>
```

Allocate parameters populated with the library defaults.

_Source: `rust/cuvs/src/neighbors/ivf_pq/params.rs:112`_

_Source: `rust/cuvs/src/neighbors/ivf_pq/params.rs:21`_

## SearchParams

```rust
pub struct SearchParams {
    /* private fields */
}
```

Parameters for searching an IVF-PQ index.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/ivf_pq/params.rs:143` |
| `try_new` | `rust/cuvs/src/neighbors/ivf_pq/params.rs:178` |

### new

```rust
#[builder]
pub fn new(
n_probes: Option<u32>,
lut_dtype: Option<LutDType>,
internal_distance_dtype: Option<InternalDistanceDType>,
) -> Result<Self, IvfPqError>
```

_Source: `rust/cuvs/src/neighbors/ivf_pq/params.rs:143`_

### try_new

```rust
pub fn try_new() -> Result<Self, IvfPqError>
```

Allocate parameters populated with the library defaults.

_Source: `rust/cuvs/src/neighbors/ivf_pq/params.rs:178`_

_Source: `rust/cuvs/src/neighbors/ivf_pq/params.rs:136`_
