---
slug: api-reference/rust-api-cuvs-cluster-kmeans-params
---

# Cluster Kmeans Params Module

_Rust module: `cuvs::cluster::kmeans::params`_

_Source: `rust/cuvs/src/cluster/kmeans/params.rs`_

Builder-pattern parameter type for k-means.

All setters are optional; unset values retain the library defaults from the
underlying C `cuvsKMeansParamsCreate`.

## Params

```rust
pub struct Params {
    /* private fields */
}
```

Parameters for k-means fitting and prediction.

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/cluster/kmeans/params.rs:29` |

### new

```rust
#[builder]
#[allow(clippy::too_many_arguments)]
pub fn new(
metric: Option<DistanceType>,
n_clusters: Option<i32>,
max_iter: Option<i32>,
tol: Option<f64>,
n_init: Option<i32>,
oversampling_factor: Option<f64>,
batch_samples: Option<i32>,
batch_centroids: Option<i32>,
hierarchical: Option<bool>,
hierarchical_n_iters: Option<i32>,
) -> Result<Self, KMeansError>
```

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:29`_

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:21`_
