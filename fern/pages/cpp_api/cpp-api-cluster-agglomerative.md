---
slug: api-reference/cpp-api-cluster-agglomerative
---

# Agglomerative

_Source header: `cuvs/cluster/agglomerative.hpp`_

## agglomerative clustering hyperparameters

<a id="cluster-agglomerative-linkage"></a>
### cluster::agglomerative::Linkage

Determines the method for computing the minimum spanning tree (MST)

```cpp
enum Linkage {
  PAIRWISE = 0,
  KNN_GRAPH = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `PAIRWISE` | `0` |
| `KNN_GRAPH` | `1` |

## Types

<a id="cluster-agglomerative-single-linkage-output"></a>
### cluster::agglomerative::single_linkage_output

Simple container object for consolidating linkage results. This closely

mirrors the trained instance variables populated in Scikit-learn's AgglomerativeClustering estimator.

```cpp
template <typename idx_t>
class single_linkage_output;
```

## single-linkage clustering APIs

<a id="cluster-agglomerative-single-linkage"></a>
### cluster::agglomerative::single_linkage

Single-linkage clustering, capable of constructing a KNN graph to

```cpp
void single_linkage(
raft::resources const& handle,
raft::device_matrix_view<const float, int, raft::row_major> X,
raft::device_matrix_view<int, int, raft::row_major> dendrogram,
raft::device_vector_view<int, int> labels,
cuvs::distance::DistanceType metric,
size_t n_clusters,
cuvs::cluster::agglomerative::Linkage linkage = cuvs::cluster::agglomerative::Linkage::KNN_GRAPH,
std::optional<int> c                          = std::make_optional<int>(DEFAULT_CONST_C));
```

scale the algorithm beyond the n^2 memory consumption of implementations that use the fully-connected graph of pairwise distances by connecting a knn graph when k is not large enough to connect it.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle |
| `X` | in | `raft::device_matrix_view<const float, int, raft::row_major>` | dense input matrix in row-major layout |
| `dendrogram` | out | `raft::device_matrix_view<int, int, raft::row_major>` | output dendrogram (size [n_rows - 1] * 2) |
| `labels` | out | `raft::device_vector_view<int, int>` | output labels vector (size n_rows) |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use when constructing connectivities graph |
| `n_clusters` | in | `size_t` | number of clusters to assign data samples |
| `linkage` | in | [`cuvs::cluster::agglomerative::Linkage`](/api-reference/cpp-api-cluster-agglomerative#cluster-agglomerative-linkage) | strategy for constructing the linkage. PAIRWISE uses more memory but can be faster for smaller datasets. KNN_GRAPH allows the memory usage to be controlled (using parameter c) at the expense of potentially additional minimum spanning tree iterations.<br />Default: `cuvs::cluster::agglomerative::Linkage::KNN_GRAPH`. |
| `c` | in | `std::optional<int>` | a constant used when constructing linkage from knn graph. Allows the indirect control of k. The algorithm will set `k = log(n) + c`<br />Default: `std::make_optional&lt;int&gt;(DEFAULT_CONST_C)`. |

**Returns**

`void`

<a id="cluster-agglomerative-helpers-linkage-graph-params-distance-params"></a>
### cluster::agglomerative::helpers::linkage_graph_params::distance_params

Specialized parameters to build the KNN graph with regular distances

```cpp
struct distance_params {
  int c;
  cuvs::cluster::agglomerative::Linkage dist_type;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `c` | `int` | a constant used when constructing linkage from knn graph. Allows the indirect control of k. The algorithm will set `k = log(n) + c` |
| `dist_type` | [`cuvs::cluster::agglomerative::Linkage`](/api-reference/cpp-api-cluster-agglomerative#cluster-agglomerative-linkage) | strategy for constructing the linkage. PAIRWISE uses more memory but can be faster for smaller datasets. KNN_GRAPH allows the memory usage to be controlled (using parameter c) |

<a id="cluster-agglomerative-helpers-linkage-graph-params-mutual-reachability-params"></a>
### cluster::agglomerative::helpers::linkage_graph_params::mutual_reachability_params

Specialized parameters to build the Mutual Reachability graph

```cpp
struct mutual_reachability_params {
  int min_samples;
  float alpha;
  cuvs::neighbors::all_neighbors::all_neighbors_params all_neighbors_params{ cuvs::neighbors::graph_build_params:: brute_force_params;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `min_samples` | `int` | this neighborhood will be selected for core distances. |
| `alpha` | `float` | weight applied when internal distance is chosen for mutual reachability (value of 1.0 disables the weighting) |
| `brute_force_params` | [`cuvs::neighbors::all_neighbors::all_neighbors_params all_neighbors_params{ cuvs::neighbors::graph_build_params::`](/api-reference/cpp-api-neighbors-all-neighbors#neighbors-all-neighbors-all-neighbors-params) | Parameters for building the mutual reachability graph using an underlying KNN algorithm.<br /><br />The all-neighbors graph construction algorithm enables building the mutual reachability graph on datasets larger than device memory by:<br />1. Partitioning the dataset into overlapping clusters,<br />2. Computing local KNN graphs within each cluster, and<br />3. Merging the local graphs into a single global graph.<br /><br />Key fields:<br />- graph_build_params: Selects the KNN construction method (Brute Force or NN Descent) and controls algorithm-specific parameters.<br /><br />- n_clusters: Number of partitions (batches) to split the data into. Larger `n_clusters` reduces memory usage but may reduce accuracy if `overlap_factor` is too low. Recommended starting value: `n_clusters = 4`. Increase progressively (4 → 8 → 16 ...) to reduce memory usage at the cost of some accuracy. This is independent of `overlap_factor` as long as `overlap_factor &lt; n_clusters`.<br /><br />- overlap_factor: Number of nearest clusters each data point is assigned to. Higher `overlap_factor` improves accuracy at the cost of memory and performance. Recommended starting value: `overlap_factor = 2`. Increase gradually (2 → 3 → 4 ...) for better accuracy with higher device memory usage.<br /><br />- metric: Distance metric to use when computing nearest neighbors. |

<a id="cluster-agglomerative-helpers-build-linkage"></a>
### cluster::agglomerative::helpers::build_linkage

Given a dataset, builds the KNN graph, connects graph components and builds a linkage

```cpp
void build_linkage(
raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> X,
std::variant<linkage_graph_params::distance_params,
linkage_graph_params::mutual_reachability_params> linkage_graph_params,
cuvs::distance::DistanceType metric,
raft::device_coo_matrix_view<float, int64_t, int64_t, size_t> out_mst,
raft::device_matrix_view<int64_t, int64_t> dendrogram,
raft::device_vector_view<float, int64_t> out_distances,
raft::device_vector_view<int64_t, int64_t> out_sizes,
std::optional<raft::device_vector_view<float, int64_t>> core_dists);
```

(dendrogram). Returns the Minimum Spanning Tree edges sorted by weight and the dendrogram.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for resource reuse |
| `X` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | data points on device memory (size n_rows * d) |
| `linkage_graph_params` | in | [`std::variant<linkage_graph_params::distance_params, linkage_graph_params::mutual_reachability_params>`](/api-reference/cpp-api-cluster-agglomerative#cluster-agglomerative-helpers-linkage-graph-params-mutual-reachability-params) | Parameters controlling how the KNN graph is built. This can be either:<br />- distance_params: standard distance-based KNN graph construction for traditional agglomerative clustering.<br />- mutual_reachability_params: parameters to compute a mutual reachability graph for density-aware hierarchical clustering (e.g. HDBSCAN). |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use |
| `out_mst` | out | `raft::device_coo_matrix_view<float, int64_t, int64_t, size_t>` | output MST sorted by edge weights (size n_rows - 1) |
| `dendrogram` | out | `raft::device_matrix_view<int64_t, int64_t>` | output dendrogram (size [n_rows - 1] * 2) |
| `out_distances` | out | `raft::device_vector_view<float, int64_t>` | distances for output |
| `out_sizes` | out | `raft::device_vector_view<int64_t, int64_t>` | cluster sizes of output |
| `core_dists` | out | `std::optional<raft::device_vector_view<float, int64_t>>` | (optional) core distances (size m). Must be supplied in the Mutual Reachability space |

**Returns**

`void`

**Additional overload:** `cluster::agglomerative::helpers::build_linkage`

Given a dataset, builds the KNN graph, connects graph components and builds a linkage

```cpp
void build_linkage(
raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> X,
std::variant<linkage_graph_params::distance_params,
linkage_graph_params::mutual_reachability_params> linkage_graph_params,
cuvs::distance::DistanceType metric,
raft::device_coo_matrix_view<float, int64_t, int64_t, size_t> out_mst,
raft::device_matrix_view<int64_t, int64_t> dendrogram,
raft::device_vector_view<float, int64_t> out_distances,
raft::device_vector_view<int64_t, int64_t> out_sizes,
std::optional<raft::device_vector_view<float, int64_t>> core_dists);
```

(dendrogram). Returns the Minimum Spanning Tree edges sorted by weight and the dendrogram.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for resource reuse |
| `X` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | data points on host memory (size n_rows * d) |
| `linkage_graph_params` | in | [`std::variant<linkage_graph_params::distance_params, linkage_graph_params::mutual_reachability_params>`](/api-reference/cpp-api-cluster-agglomerative#cluster-agglomerative-helpers-linkage-graph-params-mutual-reachability-params) | Parameters controlling how the KNN graph is built. This can be either:<br />- distance_params: standard distance-based KNN graph construction for traditional agglomerative clustering.<br />- mutual_reachability_params: parameters to compute a mutual reachability graph for density-aware hierarchical clustering (e.g. HDBSCAN). |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use |
| `out_mst` | out | `raft::device_coo_matrix_view<float, int64_t, int64_t, size_t>` | output MST sorted by edge weights (size n_rows - 1) |
| `dendrogram` | out | `raft::device_matrix_view<int64_t, int64_t>` | output dendrogram (size [n_rows - 1] * 2) |
| `out_distances` | out | `raft::device_vector_view<float, int64_t>` | distances for output |
| `out_sizes` | out | `raft::device_vector_view<int64_t, int64_t>` | cluster sizes of output |
| `core_dists` | out | `std::optional<raft::device_vector_view<float, int64_t>>` | (optional) core distances (size m). Must be supplied in the Mutual Reachability space |

**Returns**

`void`

<a id="cluster-agglomerative-helpers-build-dendrogram"></a>
### cluster::agglomerative::helpers::build_dendrogram

Build dendrogram from a Minimum Spanning Tree (MST).

```cpp
void build_dendrogram(raft::resources const& handle,
raft::device_vector_view<const int64_t, int64_t> rows,
raft::device_vector_view<const int64_t, int64_t> cols,
raft::device_vector_view<const float, int64_t> data,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> children,
raft::device_vector_view<float, int64_t> out_delta,
raft::device_vector_view<int64_t, int64_t> out_size);
```

This function takes a sorted MST (represented as edges with source, destination, and weights) and constructs a dendrogram (hierarchical clustering tree) on the host.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft resources handle |
| `rows` | in | `raft::device_vector_view<const int64_t, int64_t>` | Source nodes of the MST edges (device memory, size: nnz) |
| `cols` | in | `raft::device_vector_view<const int64_t, int64_t>` | Destination nodes of the MST edges (device memory, size: nnz) |
| `data` | in | `raft::device_vector_view<const float, int64_t>` | Edge weights/distances of the MST (device memory, size: nnz) |
| `children` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | Output dendrogram children array (device memory, size: nnz * 2) Each pair of consecutive elements represents the two children merged at each step of the hierarchy |
| `out_delta` | out | `raft::device_vector_view<float, int64_t>` | Output distances/heights at which clusters are merged (device memory, size: nnz) |
| `out_size` | out | `raft::device_vector_view<int64_t, int64_t>` | Output cluster sizes at each merge step (device memory, size: nnz) |

**Returns**

`void`
