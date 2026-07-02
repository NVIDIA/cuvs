---
slug: api-reference/cpp-api-neighbors-all-neighbors
---

# All Neighbors

_Source header: `cuvs/neighbors/all_neighbors.hpp`_

## The all-neighbors algorithm parameters.

<a id="graphbuildparams"></a>
### GraphBuildParams

The all-neighbors algorithm parameters.

```cpp
using GraphBuildParams = std::variant<graph_build_params::ivf_pq_params,
graph_build_params::nn_descent_params,
graph_build_params::brute_force_params>;
```

<a id="neighbors-all-neighbors-all-neighbors-params"></a>
### neighbors::all_neighbors::all_neighbors_params

Parameters used to build an all-neighbors graph (find nearest neighbors for all the training vectors). For scalability, the all-neighbors graph construction algorithm partitions a set of training vectors into overlapping clusters, computes a local knn graph on each cluster, and merges the local graphs into a single global graph. Device memory usage and accuracy can be configured by changing the `overlap_factor` and `n_clusters`. The algorithm used to build each local graph is also configurable.

```cpp
struct all_neighbors_params {
  GraphBuildParams graph_build_params;
  size_t overlap_factor;
  size_t n_clusters;
  cuvs::distance::DistanceType metric;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `graph_build_params` | [`GraphBuildParams`](/api-reference/cpp-api-neighbors-all-neighbors#graphbuildparams) | Parameters for knn graph building algorithm Approximate nearest neighbors methods or a brute force approach are supported to build the knn graph. Currently supported options are 'IVF-PQ', 'NN Descent', or 'Brute Force'. IVF-PQ is more accurate, but slower compared to NN Descent. Note that 'Brute Force' can also be approximate if n_clusters &gt; 1.<br /><br />Set ivf_pq_params, nn_descent_params, or brute_force_params to select the graph build algorithm and control their parameters. |
| `overlap_factor` | `size_t` | Number of nearest clusters each data point will be assigned to in the batching algorithm. Start with `overlap_factor = 2` and gradually increase (2-&gt;3-&gt;4 ...) for better accuracy at the cost of device memory usage. |
| `n_clusters` | `size_t` | Number of total clusters (aka batches) to split the data into. If set to 1, algorithm creates an all-neighbors graph without batching. Start with `n_clusters = 4` and increase (4 → 8 → 16...) for less device memory usage at the cost of accuracy. This is independent from `overlap_factor` as long as `overlap_factor` &lt; `n_clusters`.<br /><br />The ratio of `overlap_factor / n_clusters` determines device memory usage. Approximately `(overlap_factor / n_clusters) * num_rows_in_entire_data` number of rows will be put on device memory at once. E.g. between `(overlap_factor / n_clusters)` = 2/10 and 2/20, the latter will use less device memory.<br /><br />Larger `overlap_factor` results in better accuracy of the final all-neighbors knn graph. E.g. While using similar device memory, `(overlap_factor / n_clusters)` = 4/20 will have better accuracy than 2/10 at the cost of performance. |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Metric used. |

## The all-neighbors knn graph build

<a id="neighbors-all-neighbors-build"></a>
### neighbors::all_neighbors::build

Builds an approximate all-neighbors knn graph  (find nearest neighbors for all the training vectors)

```cpp
void build(
const raft::resources& handle,
const all_neighbors_params& params,
raft::host_matrix_view<const float, int64_t, row_major> dataset,
raft::device_matrix_view<int64_t, int64_t, row_major> indices,
std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances      = std::nullopt,
std::optional<raft::device_vector_view<float, int64_t, row_major>> core_distances = std::nullopt,
float alpha                                                                       = 1.0);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | raft::resources is an object managing resources |
| `params` | in | [`const all_neighbors_params&`](/api-reference/cpp-api-neighbors-all-neighbors#neighbors-all-neighbors-all-neighbors-params) | an instance of all_neighbors::all_neighbors_params that are parameters to build all-neighbors knn graph |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, row_major>` | raft::host_matrix_view input dataset expected to be located in host memory |
| `indices` | out | `raft::device_matrix_view<int64_t, int64_t, row_major>` | nearest neighbor indices of shape [n_row x k] |
| `distances` | out | `std::optional<raft::device_matrix_view<float, int64_t, row_major>>` | nearest neighbor distances [n_row x k]<br />Default: `std::nullopt`. |
| `core_distances` | out | `std::optional<raft::device_vector_view<float, int64_t, row_major>>` | array for core distances of size [n_row]. Requires distances matrix to compute core_distances. If core_distances is given, the resulting indices and distances will be mutual reachability space.<br />Default: `std::nullopt`. |
| `alpha` | in | `float` | distance scaling parameter as used in robust single linkage.<br />Default: `1.0`. |

**Returns**

`void`

**Additional overload:** `neighbors::all_neighbors::build`

Builds an approximate all-neighbors knn graph (find nearest neighbors for all the training vectors) params.n_clusters should be 1 for data on device. To use a larger params.n_clusters for efficient device memory usage, put data on host RAM.

```cpp
void build(
const raft::resources& handle,
const all_neighbors_params& params,
raft::device_matrix_view<const float, int64_t, row_major> dataset,
raft::device_matrix_view<int64_t, int64_t, row_major> indices,
std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances      = std::nullopt,
std::optional<raft::device_vector_view<float, int64_t, row_major>> core_distances = std::nullopt,
float alpha                                                                       = 1.0);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | raft::resources is an object managing resources |
| `params` | in | [`const all_neighbors_params&`](/api-reference/cpp-api-neighbors-all-neighbors#neighbors-all-neighbors-all-neighbors-params) | an instance of all_neighbors::all_neighbors_params that are parameters to build all-neighbors knn graph |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, row_major>` | raft::device_matrix_view input dataset expected to be located in device memory |
| `indices` | out | `raft::device_matrix_view<int64_t, int64_t, row_major>` | nearest neighbor indices of shape [n_row x k] |
| `distances` | out | `std::optional<raft::device_matrix_view<float, int64_t, row_major>>` | nearest neighbor distances [n_row x k]<br />Default: `std::nullopt`. |
| `core_distances` | out | `std::optional<raft::device_vector_view<float, int64_t, row_major>>` | array for core distances of size [n_row]. Requires distances matrix to compute core_distances. If core_distances is given, the resulting indices and distances will be mutual reachability space.<br />Default: `std::nullopt`. |
| `alpha` | in | `float` | distance scaling parameter as used in robust single linkage.<br />Default: `1.0`. |

**Returns**

`void`
