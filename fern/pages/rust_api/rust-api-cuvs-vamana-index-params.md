---
slug: api-reference/rust-api-cuvs-vamana-index-params
---

# Vamana Index Params Module

_Rust module: `cuvs::vamana::index_params`_

_Source: `rust/cuvs/src/vamana/index_params.rs`_

## IndexParams

```rust
pub struct IndexParams(pub ffi::cuvsVamanaIndexParams_t); {
    /* private fields */
}
```

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/vamana/index_params.rs:15` |
| `set_metric` | `rust/cuvs/src/vamana/index_params.rs:24` |
| `set_graph_degree` | `rust/cuvs/src/vamana/index_params.rs:33` |
| `set_visited_size` | `rust/cuvs/src/vamana/index_params.rs:42` |
| `set_vamana_iters` | `rust/cuvs/src/vamana/index_params.rs:50` |
| `set_alpha` | `rust/cuvs/src/vamana/index_params.rs:58` |
| `set_max_fraction` | `rust/cuvs/src/vamana/index_params.rs:67` |
| `set_batch_base` | `rust/cuvs/src/vamana/index_params.rs:75` |
| `set_queue_size` | `rust/cuvs/src/vamana/index_params.rs:83` |
| `set_reverse_batchsize` | `rust/cuvs/src/vamana/index_params.rs:91` |

### new

```rust
pub fn new() -> Result<IndexParams>
```

Returns a new IndexParams

_Source: `rust/cuvs/src/vamana/index_params.rs:15`_

### set_metric

```rust
pub fn set_metric(self, metric: DistanceType) -> IndexParams
```

DistanceType to use for building the index

_Source: `rust/cuvs/src/vamana/index_params.rs:24`_

### set_graph_degree

```rust
pub fn set_graph_degree(self, graph_degree: u32) -> IndexParams
```

Maximum degree of output graph corresponds to the R parameter in the original Vamana
literature.

_Source: `rust/cuvs/src/vamana/index_params.rs:33`_

### set_visited_size

```rust
pub fn set_visited_size(self, visited_size: u32) -> IndexParams
```

Maximum number of visited nodes per search corresponds to the L parameter in the Vamana
literature

_Source: `rust/cuvs/src/vamana/index_params.rs:42`_

### set_vamana_iters

```rust
pub fn set_vamana_iters(self, vamana_iters: f32) -> IndexParams
```

Number of Vamana vector insertion iterations (each iteration inserts all vectors).

_Source: `rust/cuvs/src/vamana/index_params.rs:50`_

### set_alpha

```rust
pub fn set_alpha(self, alpha: f32) -> IndexParams
```

Alpha for pruning parameter

_Source: `rust/cuvs/src/vamana/index_params.rs:58`_

### set_max_fraction

```rust
pub fn set_max_fraction(self, max_fraction: f32) -> IndexParams
```

Maximum fraction of dataset inserted per batch.
Larger max batch decreases graph quality, but improves speed

_Source: `rust/cuvs/src/vamana/index_params.rs:67`_

### set_batch_base

```rust
pub fn set_batch_base(self, batch_base: f32) -> IndexParams
```

Base of growth rate of batch sizes

_Source: `rust/cuvs/src/vamana/index_params.rs:75`_

### set_queue_size

```rust
pub fn set_queue_size(self, queue_size: u32) -> IndexParams
```

Size of candidate queue structure - should be (2^x)-1

_Source: `rust/cuvs/src/vamana/index_params.rs:83`_

### set_reverse_batchsize

```rust
pub fn set_reverse_batchsize(self, reverse_batchsize: u32) -> IndexParams
```

Max batchsize of reverse edge processing (reduces memory footprint)

_Source: `rust/cuvs/src/vamana/index_params.rs:91`_

_Source: `rust/cuvs/src/vamana/index_params.rs:11`_
