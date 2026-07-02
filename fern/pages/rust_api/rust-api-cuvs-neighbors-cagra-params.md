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

## CompressionParams

```rust
pub struct CompressionParams {
    /* private fields */
}
```

VPQ (Vector-Product Quantization) compression parameters.

Attach to [`IndexParams`] to enable compressed dataset storage.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:38` |
| `try_new` | `rust/cuvs/src/neighbors/cagra/params.rs:82` |

### new

```rust
#[builder]
pub fn new(
pq_bits: Option<u32>,
pq_dim: Option<u32>,
vq_n_centers: Option<u32>,
kmeans_n_iters: Option<u32>,
vq_kmeans_trainset_fraction: Option<f64>,
pq_kmeans_trainset_fraction: Option<f64>,
) -> Result<Self, CagraError>
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:38`_

### try_new

```rust
pub fn try_new() -> Result<Self, CagraError>
```

Allocate parameters populated with the library defaults.

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:82`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:31`_

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
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:130` |
| `try_new` | `rust/cuvs/src/neighbors/cagra/params.rs:194` |

### new

```rust
#[builder]
pub fn new(
metric: Option<DistanceType>,
intermediate_graph_degree: Option<usize>,
graph_degree: Option<usize>,
build_algo: Option<GraphBuildAlgo>,
nn_descent_niter: Option<usize>,
compression: Option<CompressionParams>,
) -> Result<Self, CagraError>
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:130`_

### try_new

```rust
pub fn try_new() -> Result<Self, CagraError>
```

Allocate parameters populated with the library defaults.

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:194`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:120`_

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
| `new` | `rust/cuvs/src/neighbors/cagra/params.rs:236` |
| `try_new` | `rust/cuvs/src/neighbors/cagra/params.rs:348` |

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
min_iterations: Option<usize>,
thread_block_size: Option<usize>,
hashmap_mode: Option<HashMode>,
hashmap_min_bitlen: Option<usize>,
hashmap_max_fill_rate: Option<f32>,
num_random_samplings: Option<u32>,
rand_xor_mask: Option<u64>,
) -> Result<Self, CagraError>
```

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:236`_

### try_new

```rust
pub fn try_new() -> Result<Self, CagraError>
```

Allocate parameters populated with the library defaults.

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:348`_

_Source: `rust/cuvs/src/neighbors/cagra/params.rs:228`_
