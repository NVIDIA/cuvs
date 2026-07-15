---
slug: api-reference/rust-api-cuvs-cagra-search-params
---

# Cagra Search Params Module

_Rust module: `cuvs::cagra::search_params`_

_Source: `rust/cuvs/src/cagra/search_params.rs`_

## SearchAlgo

```rust
pub type SearchAlgo = ffi::cuvsCagraSearchAlgo;
```

_Source: `rust/cuvs/src/cagra/search_params.rs:10`_

## HashMode

```rust
pub type HashMode = ffi::cuvsCagraHashMode;
```

_Source: `rust/cuvs/src/cagra/search_params.rs:11`_

## SearchParams

```rust
pub struct SearchParams(pub ffi::cuvsCagraSearchParams_t); {
    /* private fields */
}
```

Supplemental parameters to search CAGRA index

**Methods**

| Name | Source |
| --- | --- |
| `new` | `rust/cuvs/src/cagra/search_params.rs:18` |
| `set_max_queries` | `rust/cuvs/src/cagra/search_params.rs:27` |
| `set_itopk_size` | `rust/cuvs/src/cagra/search_params.rs:37` |
| `set_max_iterations` | `rust/cuvs/src/cagra/search_params.rs:45` |
| `set_algo` | `rust/cuvs/src/cagra/search_params.rs:53` |
| `set_team_size` | `rust/cuvs/src/cagra/search_params.rs:61` |
| `set_min_iterations` | `rust/cuvs/src/cagra/search_params.rs:69` |
| `set_thread_block_size` | `rust/cuvs/src/cagra/search_params.rs:77` |
| `set_hashmap_mode` | `rust/cuvs/src/cagra/search_params.rs:85` |
| `set_hashmap_min_bitlen` | `rust/cuvs/src/cagra/search_params.rs:93` |
| `set_hashmap_max_fill_rate` | `rust/cuvs/src/cagra/search_params.rs:101` |
| `set_num_random_samplings` | `rust/cuvs/src/cagra/search_params.rs:109` |
| `set_rand_xor_mask` | `rust/cuvs/src/cagra/search_params.rs:117` |

### new

```rust
pub fn new() -> Result<SearchParams>
```

Returns a new SearchParams object

_Source: `rust/cuvs/src/cagra/search_params.rs:18`_

### set_max_queries

```rust
pub fn set_max_queries(self, max_queries: usize) -> SearchParams
```

Maximum number of queries to search at the same time (batch size). Auto select when 0

_Source: `rust/cuvs/src/cagra/search_params.rs:27`_

### set_itopk_size

```rust
pub fn set_itopk_size(self, itopk_size: usize) -> SearchParams
```

Number of intermediate search results retained during the search.
This is the main knob to adjust trade off between accuracy and search speed.
Higher values improve the search accuracy

_Source: `rust/cuvs/src/cagra/search_params.rs:37`_

### set_max_iterations

```rust
pub fn set_max_iterations(self, max_iterations: usize) -> SearchParams
```

Upper limit of search iterations. Auto select when 0.

_Source: `rust/cuvs/src/cagra/search_params.rs:45`_

### set_algo

```rust
pub fn set_algo(self, algo: SearchAlgo) -> SearchParams
```

Which search implementation to use.

_Source: `rust/cuvs/src/cagra/search_params.rs:53`_

### set_team_size

```rust
pub fn set_team_size(self, team_size: usize) -> SearchParams
```

Number of threads used to calculate a single distance. 4, 8, 16, or 32.

_Source: `rust/cuvs/src/cagra/search_params.rs:61`_

### set_min_iterations

```rust
pub fn set_min_iterations(self, min_iterations: usize) -> SearchParams
```

Lower limit of search iterations.

_Source: `rust/cuvs/src/cagra/search_params.rs:69`_

### set_thread_block_size

```rust
pub fn set_thread_block_size(self, thread_block_size: usize) -> SearchParams
```

Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.

_Source: `rust/cuvs/src/cagra/search_params.rs:77`_

### set_hashmap_mode

```rust
pub fn set_hashmap_mode(self, hashmap_mode: HashMode) -> SearchParams
```

Hashmap type. Auto selection when AUTO.

_Source: `rust/cuvs/src/cagra/search_params.rs:85`_

### set_hashmap_min_bitlen

```rust
pub fn set_hashmap_min_bitlen(self, hashmap_min_bitlen: usize) -> SearchParams
```

Lower limit of hashmap bit length. More than 8.

_Source: `rust/cuvs/src/cagra/search_params.rs:93`_

### set_hashmap_max_fill_rate

```rust
pub fn set_hashmap_max_fill_rate(self, hashmap_max_fill_rate: f32) -> SearchParams
```

Upper limit of hashmap fill rate. More than 0.1, less than 0.9.

_Source: `rust/cuvs/src/cagra/search_params.rs:101`_

### set_num_random_samplings

```rust
pub fn set_num_random_samplings(self, num_random_samplings: u32) -> SearchParams
```

Number of iterations of initial random seed node selection. 1 or more.

_Source: `rust/cuvs/src/cagra/search_params.rs:109`_

### set_rand_xor_mask

```rust
pub fn set_rand_xor_mask(self, rand_xor_mask: u64) -> SearchParams
```

Bit mask used for initial random seed node selection.

_Source: `rust/cuvs/src/cagra/search_params.rs:117`_

_Source: `rust/cuvs/src/cagra/search_params.rs:14`_
