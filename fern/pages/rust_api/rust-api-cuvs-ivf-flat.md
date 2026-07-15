---
slug: api-reference/rust-api-cuvs-ivf-flat
---

# Ivf Flat Module

_Rust module: `cuvs::ivf_flat`_

_Source: `rust/cuvs/src/ivf_flat/mod.rs`_

The IVF-Flat method is an ANN algorithm. It uses an inverted file index (IVF) with
unmodified (that is, flat) vectors. This algorithm provides simple knobs to reduce
the overall search space and to trade-off accuracy for speed.

Example:
```

use cuvs::ivf_flat::{Index, IndexParams, SearchParams};
use cuvs::{ManagedTensor, Resources, Result};

use ndarray::s;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn ivf_flat_example() -> Result<()> {
let res = Resources::new()?;

// Create a new random dataset to index
let n_datapoints = 65536;
let n_features = 512;
let dataset =
ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

// build the ivf-flat index
let build_params = IndexParams::new()?;
let index = Index::build(&res, &build_params, &dataset)?;
println!(
"Indexed {}x{} datapoints into ivf-flat index",
n_datapoints, n_features
);

// use the first 4 points from the dataset as queries : will test that we get them back
// as their own nearest neighbor
let n_queries = 4;
let queries = dataset.slice(s![0..n_queries, ..]);

let k = 10;

// Ivf-Flat search API requires queries and outputs to be on device memory
// copy query data over, and allocate new device memory for the distances/ neighbors
// outputs
let queries = ManagedTensor::from(&queries).to_device(&res)?;
let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res)?;

let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
let distances = ManagedTensor::from(&distances_host).to_device(&res)?;

let search_params = SearchParams::new()?;

index.search(&res, &search_params, &queries, &neighbors, &distances)?;

// Copy back to host memory
distances.to_host(&res, &mut distances_host)?;
neighbors.to_host(&res, &mut neighbors_host)?;

// nearest neighbors should be themselves, since queries are from the
// dataset
println!("Neighbors {:?}", neighbors_host);
println!("Distances {:?}", distances_host);
Ok(())
}
```

## index::Index

```rust
pub use index::Index;
```

_Source: `rust/cuvs/src/ivf_flat/mod.rs:74`_

## index_params::IndexParams

```rust
pub use index_params::IndexParams;
```

_Source: `rust/cuvs/src/ivf_flat/mod.rs:75`_

## search_params::SearchParams

```rust
pub use search_params::SearchParams;
```

_Source: `rust/cuvs/src/ivf_flat/mod.rs:76`_
