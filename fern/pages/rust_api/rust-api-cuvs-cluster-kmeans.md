---
slug: api-reference/rust-api-cuvs-cluster-kmeans
---

# Cluster Kmeans Module

_Rust module: `cuvs::cluster::kmeans`_

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs`_

K-means clustering.

[`fit`] computes cluster centroids for a dataset, [`predict`] assigns points
to clusters, and [`cluster_cost`] reports the inertia. All inputs and outputs
reside in device memory and are borrowed through the `AsDlTensor` /
`AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack) module for the
tensor model.

## params::Params

```rust
pub use params::Params;
```

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:16`_

## KMeansError

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KMeansError {
    /* variants omitted */
}
```

Error type for k-means operations.

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:27`_

## fit

```rust
pub fn fit<X, W, C>(
res: &Resources,
params: &Params,
x: &X,
sample_weight: Option<&W>,
centroids: &mut C,
) -> Result<(f64, i32)>
where
X: AsDlTensor + ?Sized,
W: AsDlTensor + ?Sized,
C: AsDlTensorMut + ?Sized,
```

Fits k-means centroids to `x`, returning `(inertia, n_iterations)`.

`x` (shape `m × k`) is the input matrix and `centroids` (shape
`n_clusters × k`) receives the fitted centroids; `sample_weight` is an
optional per-sample weight. All reside in device memory and implement
[`AsDlTensor`] / [`AsDlTensorMut`].

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:45`_

## predict

```rust
pub fn predict<X, W, C, L>(
res: &Resources,
params: &Params,
x: &X,
sample_weight: Option<&W>,
centroids: &C,
labels: &mut L,
normalize_weight: bool,
) -> Result<f64>
where
X: AsDlTensor + ?Sized,
W: AsDlTensor + ?Sized,
C: AsDlTensor + ?Sized,
L: AsDlTensorMut + ?Sized,
```

Assigns each row of `x` to its nearest centroid, writing cluster labels into
`labels` and returning the inertia.

`x` (shape `m × k`), `centroids` (shape `n_clusters × k`), the optional
`sample_weight`, and `labels` (shape `m × 1`) reside in device memory and
implement [`AsDlTensor`] / [`AsDlTensorMut`]. `normalize_weight` selects
whether the sample weights are normalized.

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:87`_

## cluster_cost

```rust
pub fn cluster_cost<X, C>(res: &Resources, x: &X, centroids: &C) -> Result<f64>
where
X: AsDlTensor + ?Sized,
C: AsDlTensor + ?Sized,
```

Computes the k-means cost (inertia) of `x` against existing `centroids`.

`x` (shape `m × k`) and `centroids` (shape `n_clusters × k`) reside in device
memory and implement [`AsDlTensor`].

_Source: `rust/cuvs/src/cluster/kmeans/mod.rs:130`_
