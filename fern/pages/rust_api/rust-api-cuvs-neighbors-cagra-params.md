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

## CompressionParams

```rust
pub struct CompressionParams {
    /* private fields */
}
```

Parameters for VPQ compression used by CAGRA-Q.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:225` |
| `set_pq_bits` | `rust/cuvs/src/neighbors/cagra/params.rs:232` |
| `set_pq_dim` | `rust/cuvs/src/neighbors/cagra/params.rs:240` |
| `set_vq_n_centers` | `rust/cuvs/src/neighbors/cagra/params.rs:248` |
| `set_kmeans_n_iters` | `rust/cuvs/src/neighbors/cagra/params.rs:256` |
| `set_vq_kmeans_trainset_fraction` | `rust/cuvs/src/neighbors/cagra/params.rs:264` |
| `set_pq_kmeans_trainset_fraction` | `rust/cuvs/src/neighbors/cagra/params.rs:272` |

### new

```rust
pub fn new() -> Result<Self, CagraError>
```

Allocate compression params with library defaults.

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:225`_

### set_pq_bits

```rust
pub fn set_pq_bits(self, pq_bits: u32) -> Self
```

Bit length of each PQ code element. Valid values: 4..=8.

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:232`_

### set_pq_dim

```rust
pub fn set_pq_dim(self, pq_dim: u32) -> Self
```

Dimensionality after PQ compression (`0` = heuristic).

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:240`_

### set_vq_n_centers

```rust
pub fn set_vq_n_centers(self, vq_n_centers: u32) -> Self
```

VQ codebook size (`0` = heuristic).

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:248`_

### set_kmeans_n_iters

```rust
pub fn set_kmeans_n_iters(self, kmeans_n_iters: u32) -> Self
```

KMeans iterations for VQ and PQ phases.

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:256`_

### set_vq_kmeans_trainset_fraction

```rust
pub fn set_vq_kmeans_trainset_fraction(self, fraction: f64) -> Self
```

Fraction of data used for VQ kmeans (`0` = heuristic).

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:264`_

### set_pq_kmeans_trainset_fraction

```rust
pub fn set_pq_kmeans_trainset_fraction(self, fraction: f64) -> Self
```

Fraction of data used for PQ kmeans (`0` = heuristic).

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:272`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:219`_

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
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:299` |
| `set_pq_bits` | `rust/cuvs/src/neighbors/cagra/params.rs:309` |
| `set_pq_dim` | `rust/cuvs/src/neighbors/cagra/params.rs:314` |
| `set_use_subspaces` | `rust/cuvs/src/neighbors/cagra/params.rs:319` |
| `set_use_vq` | `rust/cuvs/src/neighbors/cagra/params.rs:324` |
| `set_vq_n_centers` | `rust/cuvs/src/neighbors/cagra/params.rs:329` |
| `set_kmeans_n_iters` | `rust/cuvs/src/neighbors/cagra/params.rs:334` |
| `set_max_train_points_per_pq_code` | `rust/cuvs/src/neighbors/cagra/params.rs:339` |
| `set_max_train_points_per_vq_cluster` | `rust/cuvs/src/neighbors/cagra/params.rs:344` |

### new

```rust
pub fn new() -> Result<Self, CagraError>
```

Allocate product quantizer params with library defaults.

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:299`_

### set_pq_bits

```rust
pub fn set_pq_bits(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:309`_

### set_pq_dim

```rust
pub fn set_pq_dim(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:314`_

### set_use_subspaces

```rust
pub fn set_use_subspaces(self, value: bool) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:319`_

### set_use_vq

```rust
pub fn set_use_vq(self, value: bool) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:324`_

### set_vq_n_centers

```rust
pub fn set_vq_n_centers(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:329`_

### set_kmeans_n_iters

```rust
pub fn set_kmeans_n_iters(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:334`_

### set_max_train_points_per_pq_code

```rust
pub fn set_max_train_points_per_pq_code(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:339`_

### set_max_train_points_per_vq_cluster

```rust
pub fn set_max_train_points_per_vq_cluster(self, value: u32) -> Self
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:344`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:293`_

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
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:381` |

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

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:381`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:373`_

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
