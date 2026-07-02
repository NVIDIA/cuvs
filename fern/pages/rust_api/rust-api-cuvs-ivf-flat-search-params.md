---
slug: api-reference/rust-api-cuvs-ivf-flat-search-params
---

# Ivf Flat Search Params Module

_Rust module: `cuvs::ivf_flat::search_params`_

_Source: `rust/cuvs/src/ivf_flat/search_params.rs`_

## SearchParams

```rust
pub struct SearchParams(pub ffi::cuvsIvfFlatSearchParams_t); {
    /* private fields */
}
```

Supplemental parameters to search IvfFlat index

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/ivf_flat/search_params.rs:15` |
| `set_n_probes` | `rust/cuvs/src/ivf_flat/search_params.rs:24` |

### new

```rust
pub fn new() -> Result<SearchParams>
```

Returns a new SearchParams object

_Source: `rust/cuvs/src/ivf_flat/search_params.rs:15`_

### set_n_probes

```rust
pub fn set_n_probes(self, n_probes: u32) -> SearchParams
```

Supplemental parameters to search IVF-Flat index

_Source: `rust/cuvs/src/ivf_flat/search_params.rs:24`_

_Source: `rust/cuvs/src/ivf_flat/search_params.rs:11`_
