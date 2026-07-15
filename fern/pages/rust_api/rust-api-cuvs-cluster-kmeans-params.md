---
slug: api-reference/rust-api-cuvs-cluster-kmeans-params
---

# Cluster Kmeans Params Module

_Rust module: `cuvs::cluster::kmeans::params`_

_Source: `rust/cuvs/src/cluster/kmeans/params.rs`_

## Params

```rust
pub struct Params(pub ffi::cuvsKMeansParams_t); {
    /* private fields */
}
```

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/cluster/kmeans/params.rs:15` |
| `set_metric` | `rust/cuvs/src/cluster/kmeans/params.rs:24` |
| `set_n_clusters` | `rust/cuvs/src/cluster/kmeans/params.rs:32` |
| `set_max_iter` | `rust/cuvs/src/cluster/kmeans/params.rs:40` |
| `set_tol` | `rust/cuvs/src/cluster/kmeans/params.rs:48` |
| `set_n_init` | `rust/cuvs/src/cluster/kmeans/params.rs:56` |
| `set_oversampling_factor` | `rust/cuvs/src/cluster/kmeans/params.rs:64` |
| `set_batch_samples` | `rust/cuvs/src/cluster/kmeans/params.rs:77` |
| `set_batch_centroids` | `rust/cuvs/src/cluster/kmeans/params.rs:84` |
| `set_hierarchical` | `rust/cuvs/src/cluster/kmeans/params.rs:92` |
| `set_hierarchical_n_iters` | `rust/cuvs/src/cluster/kmeans/params.rs:100` |

### new

```rust
pub fn new() -> Result<Params>
```

Returns a new Params

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:15`_

### set_metric

```rust
pub fn set_metric(self, metric: DistanceType) -> Params
```

DistanceType to use for fitting kmeans

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:24`_

### set_n_clusters

```rust
pub fn set_n_clusters(self, n_clusters: i32) -> Params
```

The number of clusters to form as well as the number of centroids to generate (default:8).

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:32`_

### set_max_iter

```rust
pub fn set_max_iter(self, max_iter: i32) -> Params
```

Maximum number of iterations of the k-means algorithm for a single run.

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:40`_

### set_tol

```rust
pub fn set_tol(self, tol: f64) -> Params
```

Relative tolerance with regards to inertia to declare convergence.

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:48`_

### set_n_init

```rust
pub fn set_n_init(self, n_init: i32) -> Params
```

Number of instance k-means algorithm will be run with different seeds.

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:56`_

### set_oversampling_factor

```rust
pub fn set_oversampling_factor(self, oversampling_factor: f64) -> Params
```

Oversampling factor for use in the k-means\|\| algorithm

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:64`_

### set_batch_samples

```rust
pub fn set_batch_samples(self, batch_samples: i32) -> Params
```

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:77`_

### set_batch_centroids

```rust
pub fn set_batch_centroids(self, batch_centroids: i32) -> Params
```

if 0 then batch_centroids = n_clusters

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:84`_

### set_hierarchical

```rust
pub fn set_hierarchical(self, hierarchical: bool) -> Params
```

Whether to use hierarchical (balanced) kmeans or not

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:92`_

### set_hierarchical_n_iters

```rust
pub fn set_hierarchical_n_iters(self, hierarchical_n_iters: i32) -> Params
```

For hierarchical k-means , defines the number of training iterations

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:100`_

_Source: `rust/cuvs/src/cluster/kmeans/params.rs:11`_
