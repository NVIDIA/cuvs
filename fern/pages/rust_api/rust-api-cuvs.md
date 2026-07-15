---
slug: api-reference/rust-api-cuvs
---

# cuVS Rust Crate

_Rust module: `cuvs`_

_Source: `rust/cuvs/src/lib.rs`_

cuVS: Rust bindings for Vector Search on the GPU

This crate provides Rust bindings for cuVS, allowing you to run
approximate nearest neighbors search on the GPU.

## brute_force

```rust
pub mod brute_force;
```

_Source: `rust/cuvs/src/lib.rs:12`_

## cagra

```rust
pub mod cagra;
```

_Source: `rust/cuvs/src/lib.rs:13`_

## cluster

```rust
pub mod cluster;
```

_Source: `rust/cuvs/src/lib.rs:14`_

## distance

```rust
pub mod distance;
```

_Source: `rust/cuvs/src/lib.rs:15`_

## distance_type

```rust
pub mod distance_type;
```

_Source: `rust/cuvs/src/lib.rs:16`_

## ivf_flat

```rust
pub mod ivf_flat;
```

_Source: `rust/cuvs/src/lib.rs:19`_

## ivf_pq

```rust
pub mod ivf_pq;
```

_Source: `rust/cuvs/src/lib.rs:20`_

## vamana

```rust
pub mod vamana;
```

_Source: `rust/cuvs/src/lib.rs:22`_

## dlpack::ManagedTensor

```rust
pub use dlpack::ManagedTensor;
```

_Source: `rust/cuvs/src/lib.rs:24`_

## error::\{Error, Result\}

```rust
pub use error::{Error, Result};
```

_Source: `rust/cuvs/src/lib.rs:25`_

## resources::Resources

```rust
pub use resources::Resources;
```

_Source: `rust/cuvs/src/lib.rs:26`_
