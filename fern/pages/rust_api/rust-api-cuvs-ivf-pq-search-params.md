---
slug: api-reference/rust-api-cuvs-ivf-pq-search-params
---

# Ivf Pq Search Params Module

_Rust module: `cuvs::ivf_pq::search_params`_

_Source: `rust/cuvs/src/ivf_pq/search_params.rs`_

## ffi::cudaDataType_t

```rust
pub use ffi::cudaDataType_t;
```

_Source: `rust/cuvs/src/ivf_pq/search_params.rs:10`_

## SearchParams

```rust
pub struct SearchParams(pub ffi::cuvsIvfPqSearchParams_t); {
    /* private fields */
}
```

Supplemental parameters to search IvfPq index

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/ivf_pq/search_params.rs:17` |
| `set_n_probes` | `rust/cuvs/src/ivf_pq/search_params.rs:26` |
| `set_lut_dtype` | `rust/cuvs/src/ivf_pq/search_params.rs:39` |
| `set_internal_distance_dtype` | `rust/cuvs/src/ivf_pq/search_params.rs:47` |

### new

```rust
pub fn new() -> Result<SearchParams>
```

Returns a new SearchParams object

_Source: `rust/cuvs/src/ivf_pq/search_params.rs:17`_

### set_n_probes

```rust
pub fn set_n_probes(self, n_probes: u32) -> SearchParams
```

The number of clusters to search.

_Source: `rust/cuvs/src/ivf_pq/search_params.rs:26`_

### set_lut_dtype

```rust
pub fn set_lut_dtype(self, lut_dtype: cudaDataType_t) -> SearchParams
```

Data type of look up table to be created dynamically at search
time. The use of low-precision types reduces the amount of shared
memory required at search time, so fast shared memory kernels can
be used even for datasets with large dimansionality. Note that
the recall is slightly degraded when low-precision type is
selected.

_Source: `rust/cuvs/src/ivf_pq/search_params.rs:39`_

### set_internal_distance_dtype

```rust
pub fn set_internal_distance_dtype(
self,
internal_distance_dtype: cudaDataType_t,
) -> SearchParams
```

Storage data type for distance/similarity computation.

_Source: `rust/cuvs/src/ivf_pq/search_params.rs:47`_

_Source: `rust/cuvs/src/ivf_pq/search_params.rs:13`_
