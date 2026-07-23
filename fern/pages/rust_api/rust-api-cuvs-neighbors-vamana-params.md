---
slug: api-reference/rust-api-cuvs-neighbors-vamana-params
---

# Neighbors Vamana Params Module

_Rust module: `cuvs::neighbors::vamana::params`_

_Source: `rust/cuvs/src/neighbors/vamana/params.rs`_

Builder-pattern parameter type for Vamana index build.

All setters are optional; unset values retain the library defaults from the
underlying C `cuvsVamanaIndexParamsCreate`.

## IndexParams

```rust
pub struct IndexParams {
    /* private fields */
}
```

Parameters for building a Vamana index.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/neighbors/vamana/params.rs:31` |

### new

```rust
#[builder]
#[allow(clippy::too_many_arguments)]
pub fn new(
metric: Option<DistanceType>,
graph_degree: Option<u32>,
visited_size: Option<u32>,
vamana_iters: Option<f32>,
alpha: Option<f32>,
max_fraction: Option<f32>,
batch_base: Option<f32>,
queue_size: Option<u32>,
reverse_batchsize: Option<u32>,
) -> Result<Self, VamanaError>
```

_Source: `rust/cuvs/src/neighbors/vamana/params.rs:31`_

_Source: `rust/cuvs/src/neighbors/vamana/params.rs:23`_
