---
slug: api-reference/rust-api-cuvs-neighbors
---

# Neighbors Module

_Rust module: `cuvs::neighbors`_

_Source: `rust/cuvs/src/neighbors/mod.rs`_

Nearest neighbor search algorithms.

Mirrors the C++ `cuvs::neighbors` namespace: each submodule wraps one index
type. Build an [`Index`](cagra::Index) from a dataset, then search it with
device-resident queries and output buffers; see the [`dlpack`](crate::dlpack)
module for the tensor model.

## brute_force

```rust
pub mod brute_force;
```

_Source: `rust/cuvs/src/neighbors/mod.rs:13`_

## cagra

```rust
pub mod cagra;
```

_Source: `rust/cuvs/src/neighbors/mod.rs:14`_

## filters

```rust
pub mod filters;
```

_Source: `rust/cuvs/src/neighbors/mod.rs:15`_

## ivf_flat

```rust
pub mod ivf_flat;
```

_Source: `rust/cuvs/src/neighbors/mod.rs:16`_

## ivf_pq

```rust
pub mod ivf_pq;
```

_Source: `rust/cuvs/src/neighbors/mod.rs:17`_

## vamana

```rust
pub mod vamana;
```

_Source: `rust/cuvs/src/neighbors/mod.rs:18`_
