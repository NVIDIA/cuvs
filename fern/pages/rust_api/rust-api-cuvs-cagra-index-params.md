---
slug: api-reference/rust-api-cuvs-cagra-index-params
---

# Cagra Index Params Module

_Rust module: `cuvs::cagra::index_params`_

_Source: `rust/cuvs/src/cagra/index_params.rs`_

## BuildAlgo

```rust
pub type BuildAlgo = ffi::cuvsCagraGraphBuildAlgo;
```

_Source: `rust/cuvs/src/cagra/index_params.rs:10`_

## CompressionParams

```rust
pub struct CompressionParams(pub ffi::cuvsCagraCompressionParams_t); {
    /* private fields */
}
```

Supplemental parameters to build CAGRA Index

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/cagra/index_params.rs:17` |
| `set_pq_bits` | `rust/cuvs/src/cagra/index_params.rs:26` |
| `set_pq_dim` | `rust/cuvs/src/cagra/index_params.rs:35` |
| `set_vq_n_centers` | `rust/cuvs/src/cagra/index_params.rs:44` |
| `set_kmeans_n_iters` | `rust/cuvs/src/cagra/index_params.rs:53` |
| `set_vq_kmeans_trainset_fraction` | `rust/cuvs/src/cagra/index_params.rs:62` |
| `set_pq_kmeans_trainset_fraction` | `rust/cuvs/src/cagra/index_params.rs:74` |

### new

```rust
pub fn new() -> Result<CompressionParams>
```

Returns a new CompressionParams

_Source: `rust/cuvs/src/cagra/index_params.rs:17`_

### set_pq_bits

```rust
pub fn set_pq_bits(self, pq_bits: u32) -> CompressionParams
```

The bit length of the vector element after compression by PQ.

_Source: `rust/cuvs/src/cagra/index_params.rs:26`_

### set_pq_dim

```rust
pub fn set_pq_dim(self, pq_dim: u32) -> CompressionParams
```

The dimensionality of the vector after compression by PQ. When zero,
an optimal value is selected using a heuristic.

_Source: `rust/cuvs/src/cagra/index_params.rs:35`_

### set_vq_n_centers

```rust
pub fn set_vq_n_centers(self, vq_n_centers: u32) -> CompressionParams
```

Vector Quantization (VQ) codebook size - number of "coarse cluster
centers". When zero, an optimal value is selected using a heuristic.

_Source: `rust/cuvs/src/cagra/index_params.rs:44`_

### set_kmeans_n_iters

```rust
pub fn set_kmeans_n_iters(self, kmeans_n_iters: u32) -> CompressionParams
```

The number of iterations searching for kmeans centers (both VQ & PQ
phases).

_Source: `rust/cuvs/src/cagra/index_params.rs:53`_

### set_vq_kmeans_trainset_fraction

```rust
pub fn set_vq_kmeans_trainset_fraction(
self,
vq_kmeans_trainset_fraction: f64,
) -> CompressionParams
```

The fraction of data to use during iterative kmeans building (VQ
phase). When zero, an optimal value is selected using a heuristic.

_Source: `rust/cuvs/src/cagra/index_params.rs:62`_

### set_pq_kmeans_trainset_fraction

```rust
pub fn set_pq_kmeans_trainset_fraction(
self,
pq_kmeans_trainset_fraction: f64,
) -> CompressionParams
```

The fraction of data to use during iterative kmeans building (PQ
phase). When zero, an optimal value is selected using a heuristic.

_Source: `rust/cuvs/src/cagra/index_params.rs:74`_

_Source: `rust/cuvs/src/cagra/index_params.rs:13`_

## IndexParams

```rust
pub struct IndexParams(pub ffi::cuvsCagraIndexParams_t, Option<CompressionParams>); {
    /* private fields */
}
```

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/cagra/index_params.rs:89` |
| `set_intermediate_graph_degree` | `rust/cuvs/src/cagra/index_params.rs:98` |
| `set_graph_degree` | `rust/cuvs/src/cagra/index_params.rs:106` |
| `set_build_algo` | `rust/cuvs/src/cagra/index_params.rs:114` |
| `set_nn_descent_niter` | `rust/cuvs/src/cagra/index_params.rs:122` |
| `set_compression` | `rust/cuvs/src/cagra/index_params.rs:129` |

### new

```rust
pub fn new() -> Result<IndexParams>
```

Returns a new IndexParams

_Source: `rust/cuvs/src/cagra/index_params.rs:89`_

### set_intermediate_graph_degree

```rust
pub fn set_intermediate_graph_degree(self, intermediate_graph_degree: usize) -> IndexParams
```

Degree of input graph for pruning

_Source: `rust/cuvs/src/cagra/index_params.rs:98`_

### set_graph_degree

```rust
pub fn set_graph_degree(self, graph_degree: usize) -> IndexParams
```

Degree of output graph

_Source: `rust/cuvs/src/cagra/index_params.rs:106`_

### set_build_algo

```rust
pub fn set_build_algo(self, build_algo: BuildAlgo) -> IndexParams
```

ANN algorithm to build knn graph

_Source: `rust/cuvs/src/cagra/index_params.rs:114`_

### set_nn_descent_niter

```rust
pub fn set_nn_descent_niter(self, nn_descent_niter: usize) -> IndexParams
```

Number of iterations to run if building with NN_DESCENT

_Source: `rust/cuvs/src/cagra/index_params.rs:122`_

### set_compression

```rust
pub fn set_compression(mut self, compression: CompressionParams) -> IndexParams
```

_Source: `rust/cuvs/src/cagra/index_params.rs:129`_

_Source: `rust/cuvs/src/cagra/index_params.rs:85`_
