---
slug: api-reference/rust-api-cuvs-neighbors-ivf-flat-params
---

# Neighbors Ivf Flat Params Module

_Rust module: `cuvs::neighbors::ivf_flat::params`_

_Source: `rust/cuvs/src/neighbors/ivf_flat/params.rs`_

Builder-pattern parameter types for IVF-Flat build and search.

All setters are optional; unset values retain the library defaults from the
underlying C `*ParamsCreate` functions.

## IndexParams

```rust
pub struct IndexParams {
    /* private fields */
}
```

Parameters for building an IVF-Flat index.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/ivf_flat/params.rs:28` |

### new

```rust
#[builder]
pub fn new(
n_lists: Option<u32>,
metric: Option<DistanceType>,
kmeans_n_iters: Option<u32>,
kmeans_trainset_fraction: Option<f64>,
add_data_on_build: Option<bool>,
) -> Result<Self, IvfFlatError>
```

_Source: `rust/cuvs/src/neighbors/ivf_flat/params.rs:28`_

_Source: `rust/cuvs/src/neighbors/ivf_flat/params.rs:21`_

## SearchParams

```rust
pub struct SearchParams {
    /* private fields */
}
```

Parameters for searching an IVF-Flat index.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/ivf_flat/params.rs:102` |

### new

```rust
#[builder]
pub fn new(n_probes: Option<u32>) -> Result<Self, IvfFlatError>
```

_Source: `rust/cuvs/src/neighbors/ivf_flat/params.rs:102`_

_Source: `rust/cuvs/src/neighbors/ivf_flat/params.rs:95`_
