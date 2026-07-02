---
slug: api-reference/rust-api-cuvs-cluster-kmeans
---

# Cluster Kmeans Module

_Rust module: `cuvs::cluster::kmeans`_

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs`_

Kmeans clustering API's

Example:
```

use cuvs::cluster::kmeans;
use cuvs::{ManagedTensor, Resources, Result};

use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn kmeans_example() -> Result<()> {
let res = Resources::new()?;

// Create a new random dataset to index
let n_datapoints = 65536;
let n_features = 512;
let n_clusters = 8;
let dataset =
ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
let dataset = ManagedTensor::from(&dataset).to_device(&res)?;

let centroids_host = ndarray::Array::<f32, _>::zeros((n_clusters, n_features));
let mut centroids = ManagedTensor::from(&centroids_host).to_device(&res)?;

// find the centroids with the kmeans index
let kmeans_params = kmeans::Params::new()?.set_n_clusters(n_clusters as i32);
let (inertia, n_iter) = kmeans::fit(&res, &kmeans_params, &dataset, &None, &mut centroids)?;
Ok(())
}
```

## params::Params

```rust
pub use params::Params;
```

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:40`_

## fit

```rust
pub fn fit(
res: &Resources,
params: &Params,
x: &ManagedTensor,
sample_weight: &Option<ManagedTensor>,
centroids: &mut ManagedTensor,
) -> Result<(f64, i32)>
```

Find clusters with the k-means algorithm

#### Arguments

* `res` - Resources to use
* `params` - Parameters to use to fit KMeans model
* `x` - A matrix in device memory - shape (m, k)
* `sample_weight` - Optional device matrix shape (n_clusters, 1)
* `centroids` - Output device matrix, that has the centroids for each cluster
shape (n_clusters, k)

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:56`_

## predict

```rust
pub fn predict(
res: &Resources,
params: &Params,
x: &ManagedTensor,
sample_weight: &Option<ManagedTensor>,
centroids: &ManagedTensor,
labels: &mut ManagedTensor,
normalize_weight: bool,
) -> Result<f64>
```

Predict clusters with the k-means algorithm

#### Arguments

* `res` - Resources to use
* `params` - Parameters to use to fit KMeans model
* `x` - Input matrix in device memory - shape (m, k)
* `sample_weight` - Optional device matrix shape (n_clusters, 1)
* `centroids` - Centroids calculated by fit in device memory, shape (n_clusters, k)
* `labels` - preallocated CUDA array interface matrix shape (m, 1) to hold the output labels
* `normalize_weight` - whether or not to normalize the weights

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:95`_

## cluster_cost

```rust
pub fn cluster_cost(res: &Resources, x: &ManagedTensor, centroids: &ManagedTensor) -> Result<f64>
```

Compute cluster cost given an input matrix and existing centroids
#### Arguments

* `res` - Resources to use
* `x` - Input matrix in device memory - shape (m, k)
* `centroids` - Centroids calculated by fit in device memory, shape (n_clusters, k)

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:131`_
