# Using NVIDIA cuVS APIs

NVIDIA cuVS provides APIs for C, C++, Python, Java, Rust, and Go. For CAGRA build and search examples in each language, see the [CAGRA indexing guide](/user-guide/api-guides/indexing-guide/cagra).

## Filtering vector indexes

NVIDIA cuVS supports different types of filtering depending on the vector index being used. The main method used by vector indexes is pre-filtering, which accounts for filtered vectors before computing closest neighbors and avoids unnecessary distance calculations.

### Bitset

A bitset is an array of bits where each bit can have two possible values: `0` and `1`. In the context of filtering, `0` means the corresponding vector is filtered out and will not be present in the search results.

Bitsets are optimized to use as little memory as possible and are available through RAFT. See the RAFT [bitset API documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitset/) for more information. When calling a search function of an ANN index, the bitset length should match the number of vectors present in the database.

For CAGRA bitset examples, see [Using Filters](/user-guide/api-guides/indexing-guide/cagra#using-filters).

### Bitmap

A bitmap is based on the same principle as a bitset, but in two dimensions. This allows users to provide a different bitset for each query being searched. See the RAFT [bitmap API documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/core_bitmap/) for more information.

For Brute-force bitmap examples, see [Using Filters](/user-guide/api-guides/indexing-guide/brute-force#using-filters).

### CAGRA filter UDF

CAGRA also supports a low-level JIT-LTO filter UDF for C++ predicates that are more naturally expressed as CUDA device code. The UDF source defines a device function that returns `true` when a source vector is allowed and `false` when it should be rejected:

```cpp
__device__ bool cuvs_filter_udf(uint32_t query_id,
                                source_index_t source_id,
                                void* filter_data);
```

`source_index_t` is currently `uint32_t` for CAGRA. `filter_data` is an opaque pointer passed through to the device predicate; if the UDF dereferences it, the pointer and any nested pointers must refer to device-accessible memory and remain valid for the duration of the search. The `query_id` passed to the UDF is the logical query id, including the batch offset when CAGRA splits a search into multiple query batches.

When `cagra::search_params::filtering_rate` is negative, CAGRA uses `filtering::udf_filter::filtering_rate`. If both are negative, CAGRA assumes `0.0` because it cannot infer UDF selectivity from the source string. Providing a realistic filtering rate helps CAGRA size its internal search work for selective filters.

Filter UDFs are candidate-validity predicates only. They receive logical query and source identifiers plus the caller-provided context pointer; they do not expose CAGRA graph traversal state, IVF probing decisions, PQ/VPQ encoded data, or other internal index layouts. NVIDIA cuVS still owns traversal, distance computation, and result selection.

Filtered CAGRA search remains approximate ANN search. The UDF prevents rejected candidates from appearing in returned results, but it does not guarantee exact brute-force filtered nearest-neighbor semantics. For CAGRA examples, see [Using Filters](/user-guide/api-guides/indexing-guide/cagra#using-filters).
