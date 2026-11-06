---
slug: api-reference/cpp-api-neighbors-cagra-params
---

# Cagra Params

_Source header: `cuvs/neighbors/cagra_params.hpp`_

## CAGRA index search parameters

<a id="neighbors-cagra-search-algo"></a>
### neighbors::cagra::search_algo

CAGRA index search parameters

```cpp
enum class search_algo {
  SINGLE_CTA = 0,
  MULTI_CTA = 1,
  MULTI_KERNEL = 2,
  AUTO = 100
};
```

**Values**

| Name | Value |
| --- | --- |
| `SINGLE_CTA` | `0` |
| `MULTI_CTA` | `1` |
| `MULTI_KERNEL` | `2` |
| `AUTO` | `100` |

## CAGRA index build parameters

<a id="neighbors-vpq-params"></a>
### neighbors::vpq_params

Parameters for VPQ compression.

```cpp
struct vpq_params {
  uint32_t pq_bits;
  uint32_t pq_dim;
  uint32_t vq_n_centers;
  uint32_t kmeans_n_iters;
  double vq_kmeans_trainset_fraction;
  double pq_kmeans_trainset_fraction;
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type;
  uint32_t max_train_points_per_pq_code;
  uint32_t max_train_points_per_vq_cluster;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ.<br /><br />Possible values: [4, 5, 6, 7, 8].<br /><br />Hint: the smaller the 'pq_bits', the smaller the index size and the better the search performance, but the lower the recall. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is selected using a heuristic.<br /><br />TODO: at the moment `dim` must be a multiple `pq_dim`. |
| `vq_n_centers` | `uint32_t` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". When zero, an optimal value is selected using a heuristic. |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (both VQ & PQ phases). |
| `vq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (VQ phase). When zero, an optimal value is selected using a heuristic. |
| `pq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (PQ phase). When zero, an optimal value is selected using a heuristic. |
| `pq_kmeans_type` | [`cuvs::cluster::kmeans::kmeans_type`](/api-reference/cpp-api-neighbors-common#cluster-kmeans-kmeans-type) | Type of k-means algorithm for PQ training. Balanced k-means tends to be faster than regular k-means for PQ training, for problem sets where the number of points per cluster are approximately equal. Regular k-means may be better for skewed cluster distributions. |
| `max_train_points_per_pq_code` | `uint32_t` | The max number of data points to use per PQ code during PQ codebook training. Using more data points per PQ code may increase the quality of PQ codebook but may also increase the build time. We will use `pq_n_centers * max_train_points_per_pq_code` training points to train each PQ codebook. |
| `max_train_points_per_vq_cluster` | `uint32_t` | The max number of data points to use per VQ cluster during training. |

<a id="graph-build-params-t"></a>
### graph_build_params_t

CAGRA index build parameters

```cpp
using graph_build_params_t = std::variant<std::monostate,
graph_build_params::ivf_pq_params,
graph_build_params::nn_descent_params,
graph_build_params::ace_params,
graph_build_params::iterative_search_params>;
```

<a id="neighbors-cagra-hnsw-heuristic-type"></a>
### neighbors::cagra::hnsw_heuristic_type

A strategy for selecting the graph build parameters based on similar HNSW index parameters.

Define how `cagra::index_params::from_hnsw_params` should construct a graph to construct a graph that is to be converted to (used by) a CPU HNSW index.

```cpp
enum class hnsw_heuristic_type : uint32_t {
  SIMILAR_SEARCH_PERFORMANCE = 0,
  SAME_GRAPH_FOOTPRINT = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `SIMILAR_SEARCH_PERFORMANCE` | `0` |
| `SAME_GRAPH_FOOTPRINT` | `1` |

<a id="neighbors-cagra-index-params-graph-build-heuristic"></a>
### neighbors::cagra::index_params::graph_build_heuristic

Select the graph build algorithm and its parameters for a dataset.

```cpp
static graph_build_params_t graph_build_heuristic(
raft::matrix_extent<int64_t> dataset,
size_t intermediate_graph_degree,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded,
size_t build_quality                = 7);
```

This is the main CAGRA build heuristic: it chooses between NN-descent and IVF-PQ based on the dataset size and tunes their parameters based on the target intermediate graph degree and the requested build quality. It returns the `graph_build_params` variant only; the caller is responsible for setting `graph_degree` / `intermediate_graph_degree`.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dataset` |  | `raft::matrix_extent<int64_t>` | The shape of the input dataset |
| `intermediate_graph_degree` |  | `size_t` | The intermediate (kNN) graph degree the build should target.<br />Note: the intermediate graph degree must be not smaller than the output graph degree; a good practice is to have it 1.5x to 2x of the desired graph_degree and a multiple of 32. |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | The distance metric to search<br />Default: `cuvs::distance::DistanceType::L2Expanded`. |
| `build_quality` |  | `size_t` | Higher values increase the build quality (and cost) up to a point. Any value is valid, but values below 20 are the most practical (default = 7).<br />Default: `7`. |

**Returns**

[`static graph_build_params_t`](/api-reference/cpp-api-neighbors-cagra-params#graph-build-params-t)

<a id="neighbors-cagra-index-params-from-dataset"></a>
### neighbors::cagra::index_params::from_dataset

Create CAGRA index parameters heuristically tuned for a dataset.

```cpp
static cagra::index_params from_dataset(
raft::matrix_extent<int64_t> dataset,
size_t graph_degree                 = 64,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded,
size_t build_quality                = 7);
```

Returns default CAGRA `index_params` with `graph_build_params` selected by `graph_build_heuristic` for the given dataset.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dataset` |  | `raft::matrix_extent<int64_t>` | The shape of the input dataset |
| `graph_degree` |  | `size_t` | Degree of the output graph.<br />Default: `64`. |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | The distance metric to search<br />Default: `cuvs::distance::DistanceType::L2Expanded`. |
| `build_quality` |  | `size_t` | Higher values increase the build quality (and cost) up to a point. Any value is valid, but values below 20 are the most practical (default = 7).<br /><br />Default: `7`. |

**Returns**

`static cagra::index_params`

<a id="neighbors-cagra-index-params-from-hnsw-params"></a>
### neighbors::cagra::index_params::from_hnsw_params

Create a CAGRA index parameters compatible with HNSW index

```cpp
static cagra::index_params from_hnsw_params(
raft::matrix_extent<int64_t> dataset,
int M,
int ef_construction,
hnsw_heuristic_type heuristic       = hnsw_heuristic_type::SIMILAR_SEARCH_PERFORMANCE,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

* IMPORTANT NOTE *

The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce exactly the same recalls and QPS for the same parameter `ef`. The graphs are different internally. Depending on the selected heuristics, the CAGRA-produced graph's QPS-Recall curve may be shifted along the curve right or left. See the heuristics descriptions for more details.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dataset` |  | `raft::matrix_extent<int64_t>` | The shape of the input dataset |
| `M` |  | `int` | HNSW index parameter M |
| `ef_construction` |  | `int` | HNSW index parameter ef_construction |
| `heuristic` |  | [`hnsw_heuristic_type`](/api-reference/cpp-api-neighbors-cagra-params#neighbors-cagra-hnsw-heuristic-type) | The heuristic to use for selecting the graph build parameters<br />Default: `hnsw_heuristic_type::SIMILAR_SEARCH_PERFORMANCE`. |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | The distance metric to search<br /><br />Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`static cagra::index_params`

## Types

<a id="neighbors-graph-build-params-iterative-search-params"></a>
### neighbors::graph_build_params::iterative_search_params

Parameters for the iterative CAGRA graph build algorithm.

Inherits from cagra::search_params so that all search tuning knobs (search_width, max_iterations, itopk_size, etc.) are available for controlling the search-and-optimize loop during graph construction. The defaults are tuned for the build loop (e.g. search_width=1, max_iterations=8) and may differ from the regular search defaults.

```cpp
struct iterative_search_params;
```

<a id="neighbors-graph-build-params-ace-params"></a>
### neighbors::graph_build_params::ace_params

Specialized parameters for ACE (Augmented Core Extraction) graph build

```cpp
struct ace_params {
  size_t npartitions;
  size_t ef_construction;
  std::string build_dir;
  bool use_disk;
  double max_host_memory_gb;
  double max_gpu_memory_gb;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `npartitions` | `size_t` | Number of partitions for ACE (Augmented Core Extraction) partitioned build.<br /><br />When set to 0 (default), the number of partitions is automatically derived based on available host and GPU memory to maximize partition size while ensuring the build fits in memory.<br /><br />Small values might improve recall but potentially degrade performance and increase memory usage. Partitions should not be too small to prevent issues in KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests).<br /><br />If the specified number of partitions results in partitions that exceed available memory, the value will be automatically increased to fit memory constraints and a warning will be issued. |
| `ef_construction` | `size_t` | The index quality for the ACE build.<br /><br />Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| `build_dir` | `std::string` | Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).<br /><br />Used when `use_disk` is true or when the graph does not fit in host and GPU memory. This should be the fastest disk in the system and hold enough space for twice the dataset, final graph, and label mapping. The directory may already exist, but ACE's named artifacts must not already exist. Simultaneous builds must use different directories. On failure, ACE removes only artifacts it created and never deletes unrelated directory contents. |
| `use_disk` | `bool` | Whether to use disk-based storage for ACE build.<br /><br />When true, enables disk-based operations for memory-efficient graph construction. |
| `max_host_memory_gb` | `double` | Maximum host memory to use for ACE build in GiB.<br /><br />When set to 0 (default), uses available host memory. When set to a positive value, limits host memory usage to the specified amount. Useful for testing or when running alongside other memory-intensive processes. |
| `max_gpu_memory_gb` | `double` | Maximum GPU memory to use for ACE build in GiB.<br /><br />When set to 0 (default), uses available GPU memory. When set to a positive value, limits GPU memory usage to the specified amount. Useful for testing or when running alongside other memory-intensive processes. |

## CAGRA index extend parameters

<a id="neighbors-cagra-extend-params"></a>
### neighbors::cagra::extend_params

CAGRA index extend parameters

```cpp
struct extend_params {
  uint32_t max_chunk_size;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `max_chunk_size` | `uint32_t` | The additional dataset is divided into chunks and added to the graph. This is the knob to adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also degrade recall because no edges are added between the nodes in the same chunk. Auto select when 0. |
