---
slug: api-reference/rust-api-cuvs-neighbors-cagra-params
---

# Neighbors Cagra Params Module

_Rust module: `cuvs::neighbors::cagra::params`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs`_

Builder-pattern parameter types for CAGRA index build and search.

Each parameter type owns its C params handle directly. The generated `bon`
builder configures that handle in the constructor, so there is no duplicate
Rust field-bag to keep in sync with the FFI state. All setters are optional;
unset values retain the library defaults from the underlying C
`*ParamsCreate` functions. Out-of-range values are rejected by `build()` with
[`CagraError::Validation`].

## IndexParams

```rust
pub struct IndexParams {
    /* private fields */
}
```

Parameters for building a CAGRA index.

```ignore
use cuvs::neighbors::cagra::IndexParams;
use cuvs::distance::DistanceType;

let params = IndexParams::builder()
.metric(DistanceType::InnerProduct)
.graph_degree(64)
.build()?;
```

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:59` |

### new

```rust
#[builder]
pub fn new(
metric: Option<DistanceType>,
intermediate_graph_degree: Option<usize>,
graph_degree: Option<usize>,
#[builder(setters(vis = "", some_fn = graph_build_internal))] graph_build: Option<
RequestedGraphBuild,
>,
) -> Result<Self, CagraError>
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:59`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:49`_

## ProductQuantizerParams

```rust
pub struct ProductQuantizerParams {
    /* private fields */
}
```

Product quantizer training parameters used to create PQ datasets.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:221` |
| `set_pq_bits` | `rust/cuvs/src/neighbors/cagra/params.rs:231` |
| `set_pq_dim` | `rust/cuvs/src/neighbors/cagra/params.rs:236` |
| `set_use_subspaces` | `rust/cuvs/src/neighbors/cagra/params.rs:241` |
| `set_use_vq` | `rust/cuvs/src/neighbors/cagra/params.rs:246` |
| `set_vq_n_centers` | `rust/cuvs/src/neighbors/cagra/params.rs:251` |
| `set_kmeans_n_iters` | `rust/cuvs/src/neighbors/cagra/params.rs:256` |
| `set_max_train_points_per_pq_code` | `rust/cuvs/src/neighbors/cagra/params.rs:261` |
| `set_max_train_points_per_vq_cluster` | `rust/cuvs/src/neighbors/cagra/params.rs:266` |

### new

```rust
pub fn new() -> Result<Self, CagraError>
```

Allocate product quantizer params with library defaults.

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:221`_

### set_pq_bits

```rust
pub fn set_pq_bits(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:231`_

### set_pq_dim

```rust
pub fn set_pq_dim(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:236`_

### set_use_subspaces

```rust
pub fn set_use_subspaces(self, value: bool) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:241`_

### set_use_vq

```rust
pub fn set_use_vq(self, value: bool) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:246`_

### set_vq_n_centers

```rust
pub fn set_vq_n_centers(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:251`_

### set_kmeans_n_iters

```rust
pub fn set_kmeans_n_iters(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:256`_

### set_max_train_points_per_pq_code

```rust
pub fn set_max_train_points_per_pq_code(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:261`_

### set_max_train_points_per_vq_cluster

```rust
pub fn set_max_train_points_per_vq_cluster(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:266`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:215`_

## SearchParams

```rust
pub struct SearchParams {
    /* private fields */
}
```

Parameters for searching a CAGRA index.

```ignore
use cuvs::neighbors::cagra::SearchParams;

let params = SearchParams::builder().itopk_size(128).build()?;
```

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:303` |

### new

```rust
#[builder]
#[allow(clippy::too_many_arguments)]
pub fn new(
max_queries: Option<usize>,
itopk_size: Option<usize>,
max_iterations: Option<usize>,
algo: Option<SearchAlgo>,
team_size: Option<usize>,
search_width: Option<usize>,
min_iterations: Option<usize>,
thread_block_size: Option<usize>,
hashmap_mode: Option<HashMode>,
hashmap_min_bitlen: Option<usize>,
hashmap_max_fill_rate: Option<f32>,
num_random_samplings: Option<u32>,
rand_xor_mask: Option<u64>,
persistent: Option<bool>,
persistent_lifetime: Option<f32>,
persistent_device_usage: Option<f32>,
) -> Result<Self, CagraError>
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:303`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:295`_

## impl IndexParamsBuilder

```rust
impl IndexParamsBuilder
```

**Methods**

| Name | Source |
| --- | --- |
| `auto` | `rust/cuvs/src/neighbors/cagra/params.rs:107` |
| `nn_descent` | `rust/cuvs/src/neighbors/cagra/params.rs:114` |
| `nn_descent_with_iterations` | `rust/cuvs/src/neighbors/cagra/params.rs:121` |
| `iterative_cagra_search` | `rust/cuvs/src/neighbors/cagra/params.rs:131` |
| `ace` | `rust/cuvs/src/neighbors/cagra/params.rs:138` |
| `ivf_pq` | `rust/cuvs/src/neighbors/cagra/params.rs:145` |

### auto

```rust
pub fn auto(self) -> IndexParamsBuilder<SetGraphBuild<S>>
where
S::GraphBuild: IsUnset,
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:107`_

### nn_descent

```rust
pub fn nn_descent(self) -> IndexParamsBuilder<SetGraphBuild<S>>
where
S::GraphBuild: IsUnset,
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:114`_

### nn_descent_with_iterations

```rust
pub fn nn_descent_with_iterations(
self,
iterations: usize,
) -> IndexParamsBuilder<SetGraphBuild<S>>
where
S::GraphBuild: IsUnset,
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:121`_

### iterative_cagra_search

```rust
pub fn iterative_cagra_search(self) -> IndexParamsBuilder<SetGraphBuild<S>>
where
S::GraphBuild: IsUnset,
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:131`_

### ace

```rust
pub fn ace(self) -> IndexParamsBuilder<SetGraphBuild<S>>
where
S::GraphBuild: IsUnset,
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:138`_

### ivf_pq

```rust
pub fn ivf_pq(self) -> IndexParamsBuilder<SetGraphBuild<S>>
where
S::GraphBuild: IsUnset,
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:145`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:106`_
