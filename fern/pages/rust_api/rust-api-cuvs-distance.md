---
slug: api-reference/rust-api-cuvs-distance
---

# Distance Module

_Rust module: `cuvs::distance`_

_Source: `rust/cuvs/src/distance/mod.rs`_

## pairwise_distance

```rust
pub fn pairwise_distance(
res: &Resources,
x: &ManagedTensor,
y: &ManagedTensor,
distances: &ManagedTensor,
metric: DistanceType,
metric_arg: Option<f32>,
) -> Result<()>
```

Compute pairwise distances between X and Y

#### Arguments

* `res` - Resources to use
* `x` - A matrix in device memory - shape (m, k)
* `y` - A matrix in device memory - shape (n, k)
* `distances` - A matrix in device memory that receives the output distances - shape (m, n)
* `metric` - DistanceType to use for building the index
* `metric_arg` - Optional value of `p` for Minkowski distances

_Source: `rust/cuvs/src/distance/mod.rs:21`_
