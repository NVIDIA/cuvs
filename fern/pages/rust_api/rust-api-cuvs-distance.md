---
slug: api-reference/rust-api-cuvs-distance
---

# Distance Module

_Rust module: `cuvs::distance`_

_Source: `rust/cuvs/src/distance/mod.rs`_

Distance metrics and pairwise distance computation.

[`DistanceType`] selects the metric used by every index and by
[`pairwise_distance`]. Inputs and output are borrowed through the
`AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
module for the tensor model.

## DistanceType

```rust
#[derive(Debug, Copy, Clone, PartialEq)]
#[non_exhaustive]
pub enum DistanceType {
    /* variants omitted */
}
```

Distance metric used for building and searching nearest neighbor indices.

_Source: `rust/cuvs/src/distance/mod.rs:22`_

## DistanceError

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DistanceError {
    /* variants omitted */
}
```

Error type for pairwise distance operations.

_Source: `rust/cuvs/src/distance/mod.rs:143`_

## pairwise_distance

```rust
pub fn pairwise_distance<X, Y, D>(
res: &Resources,
x: &X,
y: &Y,
distances: &mut D,
metric: DistanceType,
) -> Result<(), DistanceError>
where
X: AsDlTensor + ?Sized,
Y: AsDlTensor + ?Sized,
D: AsDlTensorMut + ?Sized,
```

Computes all pairwise distances between the rows of `x` (shape `m × k`) and
`y` (shape `n × k`), writing the `m × n` result into `distances`.

`x`, `y`, and `distances` reside in device memory and implement
[`AsDlTensor`] / [`AsDlTensorMut`]. `metric` selects the distance; use
[`DistanceType::LpUnexpanded`] to supply the Minkowski exponent `p` (all
other metrics use the C API default).

_Source: `rust/cuvs/src/distance/mod.rs:159`_
