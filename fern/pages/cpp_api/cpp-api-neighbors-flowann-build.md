---
slug: api-reference/cpp-api-neighbors-flowann-build
---

# Flowann Build

_Source header: `cuvs/neighbors/flowann_build.hpp`_

## Types

<a id="cuvs-neighbors-cagra-experimental-flowann-build-params"></a>
### cuvs::neighbors::cagra::experimental::flowann::build_params

Parameters for converting a CAGRA graph into FlowANN's tiered graph representation.

```cpp
struct build_params {
  std::uint16_t node_per_cacheline;
  std::uint16_t n_bits;
  std::uint32_t inner_graph_row_bytes;
  bool validate;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `node_per_cacheline` | `std::uint16_t` | Number of consecutive nodes sharing one packed inner-graph row. |
| `n_bits` | `std::uint16_t` | Number of bits used for a node offset inside its partition. |
| `inner_graph_row_bytes` | `std::uint32_t` | Packed row size in bytes. Zero selects the original average-local-degree heuristic. |
| `validate` | `bool` | Verify every encoded inner edge and every cross edge before returning. |

<a id="cuvs-neighbors-cagra-experimental-flowann-grouping-params"></a>
### cuvs::neighbors::cagra::experimental::flowann::grouping_params

Parameters controlling the locality groups used by the integrated FlowANN build.

```cpp
struct grouping_params {
  bool enabled;
  std::uint32_t n_groups;
  std::uint16_t n_bits;
  double balance_tolerance;
  std::size_t training_rows;
  std::size_t assignment_batch_rows;
  std::uint32_t kmeans_n_iters;
  bool validate;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `enabled` | `bool` | Build and store a grouped graph layout. Disable this to retain the legacy rank-only layout. |
| `n_groups` | `std::uint32_t` | Final group count. Zero uses one group if IDs fit, otherwise ceil(N * (1 + tolerance) / 2^b). Each k-means node has at most 16 children; this does not limit the final group count. |
| `n_bits` | `std::uint16_t` | Local-ID width. Zero uses min(24, max(4, align_up_4(ceil(log2(N))))). Explicit widths in [1, 32] remain available for fixed-layout builds. |
| `balance_tolerance` | `double` | Allowed relative deviation from the average group size, in the open interval `(0, 1)`. |
| `training_rows` | `std::size_t` | Maximum sampled rows per k-means node. Zero shares a 160,000-row budget by subtree leaf count, capped at 10,000 rows per child and floored at 256 rows per child. Sampling is always capped at the actual node size. |
| `assignment_batch_rows` | `std::size_t` | Number of rows assigned to groups per GPU batch. Zero targets a 256 MiB vector buffer; distance and label buffers are additional. Does not limit host capacity-repair storage. |
| `kmeans_n_iters` | `std::uint32_t` | Number of balanced k-means training iterations. |
| `validate` | `bool` | Verify every encoded edge after rearrangement. Intended for tests and small builds. |

<a id="cuvs-neighbors-cagra-experimental-flowann-index-params"></a>
### cuvs::neighbors::cagra::experimental::flowann::index_params

Parameters for the integrated rank/budget FlowANN build.

```cpp
struct index_params {
  cuvs::neighbors::cagra::index_params cagra_params;
  std::size_t device_graph_budget_bytes;
  std::uint16_t node_per_cacheline;
  grouping_params grouping;
  std::uint32_t num_seeds;
  std::size_t seed_training_rows;
  std::uint64_t seed;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `cagra_params` | [`cuvs::neighbors::cagra::index_params`](/api-reference/cpp-api-neighbors-flowann-build#cuvs-neighbors-cagra-experimental-flowann-index-params) | Parameters used to build the complete CAGRA graph before tiering. |
| `device_graph_budget_bytes` | `std::size_t` | Maximum bytes occupied by the final packed resident graph. Grouped builds derive uniform row capacity from this budget, capped at the full graph degree. Zero stores all edges on host. |
| `node_per_cacheline` | `std::uint16_t` | Number of consecutive nodes sharing one packed inner-graph row. |
| `grouping` | [`grouping_params`](/api-reference/cpp-api-neighbors-flowann-build#cuvs-neighbors-cagra-experimental-flowann-grouping-params) | Locality grouping and group-local ID compression parameters. |
| `num_seeds` | `std::uint32_t` | Number of medoid seeds generated during build. Zero disables seed generation. |
| `seed_training_rows` | `std::size_t` | Number of sampled rows used to train seed centroids. Zero selects an automatic limit. |
| `seed` | `std::uint64_t` | Deterministic seed used to rotate the uniform k-means training sample. |

<a id="cuvs-neighbors-cagra-experimental-flowann-grouping-result"></a>
### cuvs::neighbors::cagra::experimental::flowann::grouping_result

Grouping output, reusable when the complete CAGRA graph is already available.

```cpp
struct grouping_result {
  raft::host_vector<std::uint32_t, int64_t> labels;
  std::uint16_t n_bits;
  std::uint32_t n_groups;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `labels` | `raft::host_vector<std::uint32_t, int64_t>` | One group ID in [0, n_groups) per original dataset row; owns its host storage. |
| `n_bits` | `std::uint16_t` | Resolved uniform local-ID width, including automatic width selection. |
| `n_groups` | `std::uint32_t` | Resolved final group count. |
