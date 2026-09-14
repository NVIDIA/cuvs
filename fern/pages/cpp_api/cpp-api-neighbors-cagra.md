---
slug: api-reference/cpp-api-neighbors-cagra
---

# Cagra

_Source header: `cuvs/neighbors/cagra.hpp`_

## Types

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
| `pq_kmeans_type` | [`cuvs::cluster::kmeans::kmeans_type`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-kmeans-type) | Type of k-means algorithm for PQ training. Balanced k-means tends to be faster than regular k-means for PQ training, for problem sets where the number of points per cluster are approximately equal. Regular k-means may be better for skewed cluster distributions. |
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
enum class hnsw_heuristic_type : uint32_t;
```

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

[`static graph_build_params_t`](/api-reference/cpp-api-neighbors-cagra#graph-build-params-t)

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
| `heuristic` |  | [`hnsw_heuristic_type`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-hnsw-heuristic-type) | The heuristic to use for selecting the graph build parameters<br />Default: `hnsw_heuristic_type::SIMILAR_SEARCH_PERFORMANCE`. |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | The distance metric to search<br /><br />Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`static cagra::index_params`

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

## CAGRA index type

<a id="neighbors-cagra-index"></a>
### neighbors::cagra::index

CAGRA index.

The index stores the dataset and a kNN graph in device memory.

```cpp
template <typename T,
typename IdxT,
ann_dataset_view DatasetViewT = device_padded_dataset_view<T, int64_t>>
struct index;
```

<a id="neighbors-cagra-index-metric"></a>
### neighbors::cagra::index::metric

Distance metric used for clustering.

```cpp
[[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="neighbors-cagra-index-size"></a>
### neighbors::cagra::index::size

Total length of the index (number of vectors).

```cpp
[[nodiscard]] constexpr inline auto size() const noexcept -> IdxT;
```

**Returns**

`IdxT`

<a id="neighbors-cagra-index-dim"></a>
### neighbors::cagra::index::dim

Dimensionality of the data.

```cpp
[[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="neighbors-cagra-index-graph-degree"></a>
### neighbors::cagra::index::graph_degree

Graph degree

```cpp
[[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="neighbors-cagra-index-graph-size"></a>
### neighbors::cagra::index::graph_size

Number of rows represented by the graph.

```cpp
[[nodiscard]] constexpr inline auto graph_size() const noexcept -> IdxT;
```

**Returns**

`IdxT`

<a id="neighbors-cagra-index-dataset"></a>
### neighbors::cagra::index::dataset

Non-owning dataset binding stored by the index.

```cpp
[[nodiscard]] inline auto dataset() const noexcept -> DatasetViewT const&;
```

**Returns**

`DatasetViewT const&`

<a id="neighbors-cagra-index-graph"></a>
### neighbors::cagra::index::graph

neighborhood graph [size, graph-degree]

```cpp
[[nodiscard]] inline auto graph() const noexcept
-> raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major>;
```

**Returns**

`raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major>`

<a id="neighbors-cagra-index-source-indices"></a>
### neighbors::cagra::index::source_indices

Mapping from internal graph node indices to the original user-provided indices.

```cpp
[[nodiscard]] inline auto source_indices() const noexcept
-> std::optional<raft::device_vector_view<const index_type, int64_t>>;
```

**Returns**

`std::optional<raft::device_vector_view<const index_type, int64_t>>`

<a id="neighbors-cagra-index-dataset-fd"></a>
### neighbors::cagra::index::dataset_fd

Get the dataset file descriptor (for disk-backed index)

```cpp
[[nodiscard]] inline auto dataset_fd() const noexcept
-> const std::optional<cuvs::util::file_descriptor>&;
```

**Returns**

[`const std::optional<cuvs::util::file_descriptor>&`](/api-reference/cpp-api-util-file-io#util-file-descriptor)

<a id="neighbors-cagra-index-graph-fd"></a>
### neighbors::cagra::index::graph_fd

Get the graph file descriptor (for disk-backed index)

```cpp
[[nodiscard]] inline auto graph_fd() const noexcept
-> const std::optional<cuvs::util::file_descriptor>&;
```

**Returns**

[`const std::optional<cuvs::util::file_descriptor>&`](/api-reference/cpp-api-util-file-io#util-file-descriptor)

<a id="neighbors-cagra-index-mapping-fd"></a>
### neighbors::cagra::index::mapping_fd

Get the mapping file descriptor (for disk-backed index)

```cpp
[[nodiscard]] inline auto mapping_fd() const noexcept
-> const std::optional<cuvs::util::file_descriptor>&;
```

**Returns**

[`const std::optional<cuvs::util::file_descriptor>&`](/api-reference/cpp-api-util-file-io#util-file-descriptor)

<a id="neighbors-cagra-index-dataset-norms"></a>
### neighbors::cagra::index::dataset_norms

Dataset norms for cosine distance [size]

```cpp
[[nodiscard]] inline auto dataset_norms() const noexcept
-> std::optional<raft::device_vector_view<const float, int64_t>>;
```

**Returns**

`std::optional<raft::device_vector_view<const float, int64_t>>`

<a id="neighbors-cagra-index-index"></a>
### neighbors::cagra::index::index

```cpp
index(const index&)                    = delete;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`const index&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::index::index`

Construct a graph-only index with a zero-row dataset view placeholder.

```cpp
explicit index(raft::resources const& res,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
requires(cuvs::neighbors::ann_dataset_view<DatasetViewT, int64_t>)
: cuvs::neighbors::index(),;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`explicit`

**Additional overload:** `neighbors::cagra::index::index`

Construct an index from a `dataset_view` and knn_graph.

```cpp
template <typename graph_accessor>
index(raft::resources const& res,
cuvs::distance::DistanceType metric,
DatasetViewT const& dataset,
raft::mdspan<const graph_index_type,
raft::matrix_extent<int64_t>,
raft::row_major,
graph_accessor> knn_graph);
```

Stores a shallow copy of the dataset view. The index stores a **non-owning** view; the caller must keep the underlying host or device storage alive for the index lifetime.

Example — **non-owning** `make_device_padded_dataset_view` (wraps an existing device matrix; that matrix must outlive the index):

Example — **owning** `make_device_padded_dataset` returns owning storage (`std::unique_ptr`). You must **keep that object alive** (e.g. hold the `unique_ptr` in a variable or member) for as long as the index uses the dataset; the index does not take ownership of the buffer.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `dataset` |  | `DatasetViewT const&` |  |
| `knn_graph` |  | `raft::mdspan<const graph_index_type, raft::matrix_extent<int64_t>, raft::row_major, graph_accessor>` |  |

**Returns**

`void`

<a id="neighbors-cagra-index-update-graph"></a>
### neighbors::cagra::index::update_graph

Replace the graph with a new graph.

```cpp
void update_graph(
raft::resources const& res,
raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major> knn_graph);
```

Since the new graph is a device array, we store a reference to that, and it is the caller's responsibility to ensure that knn_graph stays alive as long as the index.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `knn_graph` |  | `raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::index::update_graph`

Replace the graph by taking ownership of an existing device matrix.

```cpp
void update_graph(raft::resources const&,
raft::device_matrix<graph_index_type, int64_t, raft::row_major>&& knn_graph);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `raft::resources const&` |  |
| `knn_graph` |  | `raft::device_matrix<graph_index_type, int64_t, raft::row_major>&&` |  |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::index::update_graph`

Replace the graph with a new graph.

```cpp
void update_graph(
raft::resources const& res,
raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major> knn_graph);
```

We create a copy of the graph on the device. The index manages the lifetime of this copy.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `knn_graph` |  | `raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major>` |  |

**Returns**

`void`

<a id="neighbors-cagra-index-update-source-indices"></a>
### neighbors::cagra::index::update_source_indices

Replace the source indices with a new source indices taking the ownership of the passed vector.

```cpp
void update_source_indices(raft::device_vector<index_type, int64_t>&& source_indices);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `source_indices` |  | `raft::device_vector<index_type, int64_t>&&` |  |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::index::update_source_indices`

Copy the provided source indices into the index.

```cpp
template <typename Accessor>
void update_source_indices(
raft::resources const& res,
raft::mdspan<const index_type, raft::vector_extent<int64_t>, raft::row_major, Accessor>
source_indices);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `source_indices` |  | `raft::mdspan<const index_type, raft::vector_extent<int64_t>, raft::row_major, Accessor>` |  |

**Returns**

`void`

<a id="neighbors-cagra-index-update-dataset"></a>
### neighbors::cagra::index::update_dataset

Update the dataset from a disk file using a file descriptor.

```cpp
void update_dataset(raft::resources const& res, cuvs::util::file_descriptor&& fd);
```

This method configures the index to use a disk-based dataset. The dataset file should contain a numpy header followed by vectors in row-major format. The number of rows and dimensionality are read from the numpy header.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `fd` | in | [`cuvs::util::file_descriptor&&`](/api-reference/cpp-api-util-file-io#util-file-descriptor) | File descriptor (will be moved into the index for lifetime management) |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::index::update_graph`

Update the graph from a disk file using a file descriptor.

```cpp
void update_graph(raft::resources const& res, cuvs::util::file_descriptor&& fd);
```

This method configures the index to use a disk-based graph. The graph file should contain a numpy header followed by neighbor indices in row-major format. The number of rows and graph degree are read from the numpy header.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `fd` | in | [`cuvs::util::file_descriptor&&`](/api-reference/cpp-api-util-file-io#util-file-descriptor) | File descriptor (will be moved into the index for lifetime management) |

**Returns**

`void`

<a id="neighbors-cagra-index-update-mapping"></a>
### neighbors::cagra::index::update_mapping

Update the dataset mapping from a disk file using a file descriptor.

```cpp
void update_mapping(raft::resources const& res, cuvs::util::file_descriptor&& fd);
```

This method configures the index to use a disk-based dataset mapping. The mapping file should contain a numpy header followed by index mappings.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `fd` | in | [`cuvs::util::file_descriptor&&`](/api-reference/cpp-api-util-file-io#util-file-descriptor) | File descriptor (will be moved into the index for lifetime management) |

**Returns**

`void`

## CAGRA index build functions

<a id="neighbors-cagra-build"></a>
### neighbors::cagra::build

Build from a device padded dataset view (`float`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::device_padded_dataset_view<float, int64_t> const& dataset)
-> cuvs::neighbors::cagra::device_padded_index<float, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::device_padded_dataset_view<float, int64_t> const&` | device padded dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::device_padded_index<float, uint32_t>`

built `device_padded_index&lt;float, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a device standard dataset view (`float`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::device_standard_dataset_view<float, int64_t> const& dataset)
-> cuvs::neighbors::cagra::device_standard_index<float, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::device_standard_dataset_view<float, int64_t> const&` | device standard dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::device_standard_index<float, uint32_t>`

built `device_standard_index&lt;float, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a host padded dataset view (`float`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::host_padded_dataset_view<float, int64_t> const& dataset)
-> cuvs::neighbors::cagra::host_padded_index<float, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::host_padded_dataset_view<float, int64_t> const&` | host padded dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::host_padded_index<float, uint32_t>`

built `host_padded_index&lt;float, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a host standard dataset view (`float`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::host_standard_dataset_view<float, int64_t> const& dataset)
-> cuvs::neighbors::cagra::host_standard_index<float, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::host_standard_dataset_view<float, int64_t> const&` | host standard dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::host_standard_index<float, uint32_t>`

built `host_standard_index&lt;float, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a device padded dataset view (`half`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::device_padded_dataset_view<half, int64_t> const& dataset)
-> cuvs::neighbors::cagra::device_padded_index<half, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::device_padded_dataset_view<half, int64_t> const&` | device padded dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::device_padded_index<half, uint32_t>`

built `device_padded_index&lt;half, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a device standard dataset view (`half`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::device_standard_dataset_view<half, int64_t> const& dataset)
-> cuvs::neighbors::cagra::device_standard_index<half, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::device_standard_dataset_view<half, int64_t> const&` | device standard dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::device_standard_index<half, uint32_t>`

built `device_standard_index&lt;half, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a host padded dataset view (`half`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::host_padded_dataset_view<half, int64_t> const& dataset)
-> cuvs::neighbors::cagra::host_padded_index<half, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::host_padded_dataset_view<half, int64_t> const&` | host padded dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::host_padded_index<half, uint32_t>`

built `host_padded_index&lt;half, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a host standard dataset view (`half`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::host_standard_dataset_view<half, int64_t> const& dataset)
-> cuvs::neighbors::cagra::host_standard_index<half, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::host_standard_dataset_view<half, int64_t> const&` | host standard dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::host_standard_index<half, uint32_t>`

built `host_standard_index&lt;half, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a device padded dataset view (`int8_t`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t> const& dataset)
-> cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t> const&` | device padded dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>`

built `device_padded_index&lt;int8_t, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a device standard dataset view (`int8_t`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::device_standard_dataset_view<int8_t, int64_t> const& dataset)
-> cuvs::neighbors::cagra::device_standard_index<int8_t, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::device_standard_dataset_view<int8_t, int64_t> const&` | device standard dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::device_standard_index<int8_t, uint32_t>`

built `device_standard_index&lt;int8_t, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a host padded dataset view (`int8_t`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::host_padded_dataset_view<int8_t, int64_t> const& dataset)
-> cuvs::neighbors::cagra::host_padded_index<int8_t, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::host_padded_dataset_view<int8_t, int64_t> const&` | host padded dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::host_padded_index<int8_t, uint32_t>`

built `host_padded_index&lt;int8_t, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a host standard dataset view (`int8_t`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::host_standard_dataset_view<int8_t, int64_t> const& dataset)
-> cuvs::neighbors::cagra::host_standard_index<int8_t, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::host_standard_dataset_view<int8_t, int64_t> const&` | host standard dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::host_standard_index<int8_t, uint32_t>`

built `host_standard_index&lt;int8_t, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a device padded dataset view (`uint8_t`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t> const& dataset)
-> cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t> const&` | device padded dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>`

built `device_padded_index&lt;uint8_t, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a device standard dataset view (`uint8_t`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::device_standard_dataset_view<uint8_t, int64_t> const& dataset)
-> cuvs::neighbors::cagra::device_standard_index<uint8_t, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::device_standard_dataset_view<uint8_t, int64_t> const&` | device standard dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::device_standard_index<uint8_t, uint32_t>`

built `device_standard_index&lt;uint8_t, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a host padded dataset view (`uint8_t`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::host_padded_dataset_view<uint8_t, int64_t> const& dataset)
-> cuvs::neighbors::cagra::host_padded_index<uint8_t, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::host_padded_dataset_view<uint8_t, int64_t> const&` | host padded dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::host_padded_index<uint8_t, uint32_t>`

built `host_padded_index&lt;uint8_t, uint32_t&gt;`

**Additional overload:** `neighbors::cagra::build`

Build from a host standard dataset view (`uint8_t`).

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
cuvs::neighbors::host_standard_dataset_view<uint8_t, int64_t> const& dataset)
-> cuvs::neighbors::cagra::host_standard_index<uint8_t, uint32_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | CAGRA index build parameters |
| `dataset` | in | `cuvs::neighbors::host_standard_dataset_view<uint8_t, int64_t> const&` | host standard dataset view [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::host_standard_index<uint8_t, uint32_t>`

built `host_standard_index&lt;uint8_t, uint32_t&gt;`

## CAGRA extend functions

<a id="neighbors-cagra-extend"></a>
### neighbors::cagra::extend

Add new vectors to a CAGRA index

```cpp
void extend(raft::resources const& handle,
const cagra::extend_params& params,
cuvs::neighbors::device_padded_dataset_view<float, int64_t> extended_dataset,
int64_t new_start_row,
cuvs::neighbors::cagra::device_padded_index<float, uint32_t>& idx);
```

Note: `extend` does not concatenate datasets. The caller owns the final dataset and must pre-populate a single padded device matrix of size `(n_old + n_new) x dim` (or overallocation with a view whose logical `n_rows` is `n_old + n_new`):

- rows `[0, new_start_row)` hold the original vectors attached to `idx`
- rows `[new_start_row, n_rows)` hold the additional vectors `new_start_row` must equal `idx.size()` today. The library only extends the graph and rebinds the index to `extended_dataset`. Keep that view alive for the index lifetime.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-extend-params) | extend params |
| `extended_dataset` | in | `cuvs::neighbors::device_padded_dataset_view<float, int64_t>` | caller-owned device-padded view already containing old \|\| new rows |
| `new_start_row` | in | `int64_t` | row index where the additional vectors begin (must equal `idx.size()`) |
| `idx` | in,out | `cuvs::neighbors::cagra::device_padded_index<float, uint32_t>&` | CAGRA index; graph is extended and dataset view is rebound |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::extend`

Add new vectors to a CAGRA index. See the float overload for the full contract.

```cpp
void extend(raft::resources const& handle,
const cagra::extend_params& params,
cuvs::neighbors::device_padded_dataset_view<half, int64_t> extended_dataset,
int64_t new_start_row,
cuvs::neighbors::cagra::device_padded_index<half, uint32_t>& idx);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-extend-params) |  |
| `extended_dataset` |  | `cuvs::neighbors::device_padded_dataset_view<half, int64_t>` |  |
| `new_start_row` |  | `int64_t` |  |
| `idx` |  | `cuvs::neighbors::cagra::device_padded_index<half, uint32_t>&` |  |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::extend`

Add new vectors to a CAGRA index. See the float overload for the full contract.

```cpp
void extend(raft::resources const& handle,
const cagra::extend_params& params,
cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t> extended_dataset,
int64_t new_start_row,
cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>& idx);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-extend-params) |  |
| `extended_dataset` |  | `cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t>` |  |
| `new_start_row` |  | `int64_t` |  |
| `idx` |  | `cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>&` |  |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::extend`

Add new vectors to a CAGRA index. See the float overload for the full contract.

```cpp
void extend(raft::resources const& handle,
const cagra::extend_params& params,
cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t> extended_dataset,
int64_t new_start_row,
cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>& idx);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-extend-params) |  |
| `extended_dataset` |  | `cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t>` |  |
| `new_start_row` |  | `int64_t` |  |
| `idx` |  | `cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>&` |  |

**Returns**

`void`

## CAGRA serialize functions

<a id="neighbors-cagra-serialized-dataset-kind"></a>
### neighbors::cagra::serialized_dataset_kind

Dense dataset storage kind recorded in a serialized CAGRA index.

```cpp
enum class serialized_dataset_kind : std::uint32_t {
  none = 0,
  device_padded = 1,
  device_standard = 2,
  host_padded = 3,
  host_standard = 4
};
```

**Values**

| Name | Value |
| --- | --- |
| `none` | `0` |
| `device_padded` | `1` |
| `device_standard` | `2` |
| `host_padded` | `3` |
| `host_standard` | `4` |

<a id="neighbors-cagra-serialize"></a>
### neighbors::cagra::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::device_padded_index<float>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<float>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

<a id="neighbors-cagra-deserialize"></a>
### neighbors::cagra::deserialize

Load index from file.

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::device_padded_index<float>* index,
std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>>* out_dataset = nullptr);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::cagra::device_padded_index<float>*` | the cagra index |
| `out_dataset` | out | `std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>>*` | if non-null, on success may be set to an owned deserialized dataset when the file includes dataset data; may be left unchanged otherwise. Optional; pass nullptr to ignore.<br />Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::device_padded_index<float>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<float>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

Load index from input stream

```cpp
void deserialize(
raft::resources const& handle,
std::istream& is,
cuvs::neighbors::cagra::device_padded_index<float>* index,
std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>>* out_dataset = nullptr);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::cagra::device_padded_index<float>*` | the cagra index |
| `out_dataset` | out | `std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>>*` | if non-null, on success may be set to an owned deserialized dataset when the stream includes dataset data; may be left unchanged otherwise. Optional; pass nullptr to ignore.<br />Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::device_padded_index<half>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<half>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

Load index from file.

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::device_padded_index<half>* index,
std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>>* out_dataset = nullptr);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::cagra::device_padded_index<half>*` | the cagra index |
| `out_dataset` | out | `std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>>*` | if non-null, on success may be set to an owned deserialized dataset when the file includes dataset data; may be left unchanged otherwise. Optional; pass nullptr to ignore.<br />Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::device_padded_index<half>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<half>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

Load index from input stream

```cpp
void deserialize(
raft::resources const& handle,
std::istream& is,
cuvs::neighbors::cagra::device_padded_index<half>* index,
std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>>* out_dataset = nullptr);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::cagra::device_padded_index<half>*` | the cagra index |
| `out_dataset` | out | `std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>>*` | if non-null, on success may be set to an owned deserialized dataset when the stream includes dataset data; may be left unchanged otherwise. Optional; pass nullptr to ignore.<br />Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::device_padded_index<int8_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<int8_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

Load index from file.

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::device_padded_index<int8_t>* index,
std::unique_ptr<cuvs::neighbors::device_padded_dataset<int8_t, int64_t>>* out_dataset = nullptr);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::cagra::device_padded_index<int8_t>*` | the cagra index |
| `out_dataset` | out | `std::unique_ptr<cuvs::neighbors::device_padded_dataset<int8_t, int64_t>>*` | if non-null, on success may be set to an owned deserialized dataset when the file includes dataset data; may be left unchanged otherwise. Optional; pass nullptr to ignore.<br />Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::device_padded_index<int8_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<int8_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

Load index from input stream

```cpp
void deserialize(
raft::resources const& handle,
std::istream& is,
cuvs::neighbors::cagra::device_padded_index<int8_t>* index,
std::unique_ptr<cuvs::neighbors::device_padded_dataset<int8_t, int64_t>>* out_dataset = nullptr);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::cagra::device_padded_index<int8_t>*` | the cagra index |
| `out_dataset` | out | `std::unique_ptr<cuvs::neighbors::device_padded_dataset<int8_t, int64_t>>*` | if non-null, on success may be set to an owned deserialized dataset when the stream includes dataset data; may be left unchanged otherwise. Optional; pass nullptr to ignore.<br />Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::device_padded_index<uint8_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<uint8_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

Load index from file.

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::device_padded_index<uint8_t>* index,
std::unique_ptr<cuvs::neighbors::device_padded_dataset<uint8_t, int64_t>>* out_dataset = nullptr);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::cagra::device_padded_index<uint8_t>*` | the cagra index |
| `out_dataset` | out | `std::unique_ptr<cuvs::neighbors::device_padded_dataset<uint8_t, int64_t>>*` | if non-null, on success may be set to an owned deserialized dataset when the file includes dataset data; may be left unchanged otherwise. Optional; pass nullptr to ignore.<br />Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::device_padded_index<uint8_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<uint8_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

Load index from input stream

```cpp
void deserialize(
raft::resources const& handle,
std::istream& is,
cuvs::neighbors::cagra::device_padded_index<uint8_t>* index,
std::unique_ptr<cuvs::neighbors::device_padded_dataset<uint8_t, int64_t>>* out_dataset = nullptr);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::cagra::device_padded_index<uint8_t>*` | the cagra index |
| `out_dataset` | out | `std::unique_ptr<cuvs::neighbors::device_padded_dataset<uint8_t, int64_t>>*` | if non-null, on success may be set to an owned deserialized dataset when the stream includes dataset data; may be left unchanged otherwise. Optional; pass nullptr to ignore.<br />Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::host_padded_index<float>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<float>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_padded_index<float>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<float>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::host_standard_index<float>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<float>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_standard_index<float>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<float>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::host_padded_index<half>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<half>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_padded_index<half>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<half>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::host_standard_index<half>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<half>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_standard_index<half>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<half>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::host_padded_index<int8_t>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<int8_t>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_padded_index<int8_t>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<int8_t>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::host_standard_index<int8_t>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<int8_t>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_standard_index<int8_t>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<int8_t>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::host_padded_index<uint8_t>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<uint8_t>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_padded_index<uint8_t>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<uint8_t>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::host_standard_index<uint8_t>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<uint8_t>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize`

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_standard_index<uint8_t>& index,
bool include_dataset = true);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<uint8_t>&` |  |
| `include_dataset` |  | `bool` | Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::host_padded_index<float>* index,
std::unique_ptr<cuvs::neighbors::host_padded_dataset<float, int64_t>>* out_dataset = nullptr);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `cuvs::neighbors::cagra::host_padded_index<float>*` |  |
| `out_dataset` |  | `std::unique_ptr<cuvs::neighbors::host_padded_dataset<float, int64_t>>*` | Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::host_standard_index<float>* index,
std::unique_ptr<cuvs::neighbors::host_standard_dataset<float, int64_t>>* out_dataset = nullptr);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `cuvs::neighbors::cagra::host_standard_index<float>*` |  |
| `out_dataset` |  | `std::unique_ptr<cuvs::neighbors::host_standard_dataset<float, int64_t>>*` | Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::host_padded_index<half>* index,
std::unique_ptr<cuvs::neighbors::host_padded_dataset<half, int64_t>>* out_dataset = nullptr);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `cuvs::neighbors::cagra::host_padded_index<half>*` |  |
| `out_dataset` |  | `std::unique_ptr<cuvs::neighbors::host_padded_dataset<half, int64_t>>*` | Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::host_standard_index<half>* index,
std::unique_ptr<cuvs::neighbors::host_standard_dataset<half, int64_t>>* out_dataset = nullptr);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `cuvs::neighbors::cagra::host_standard_index<half>*` |  |
| `out_dataset` |  | `std::unique_ptr<cuvs::neighbors::host_standard_dataset<half, int64_t>>*` | Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::host_padded_index<int8_t>* index,
std::unique_ptr<cuvs::neighbors::host_padded_dataset<int8_t, int64_t>>* out_dataset = nullptr);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `cuvs::neighbors::cagra::host_padded_index<int8_t>*` |  |
| `out_dataset` |  | `std::unique_ptr<cuvs::neighbors::host_padded_dataset<int8_t, int64_t>>*` | Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::host_standard_index<int8_t>* index,
std::unique_ptr<cuvs::neighbors::host_standard_dataset<int8_t, int64_t>>* out_dataset = nullptr);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `cuvs::neighbors::cagra::host_standard_index<int8_t>*` |  |
| `out_dataset` |  | `std::unique_ptr<cuvs::neighbors::host_standard_dataset<int8_t, int64_t>>*` | Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::host_padded_index<uint8_t>* index,
std::unique_ptr<cuvs::neighbors::host_padded_dataset<uint8_t, int64_t>>* out_dataset = nullptr);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `cuvs::neighbors::cagra::host_padded_index<uint8_t>*` |  |
| `out_dataset` |  | `std::unique_ptr<cuvs::neighbors::host_padded_dataset<uint8_t, int64_t>>*` | Default: `nullptr`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::deserialize`

```cpp
void deserialize(
raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::host_standard_index<uint8_t>* index,
std::unique_ptr<cuvs::neighbors::host_standard_dataset<uint8_t, int64_t>>* out_dataset = nullptr);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | `cuvs::neighbors::cagra::host_standard_index<uint8_t>*` |  |
| `out_dataset` |  | `std::unique_ptr<cuvs::neighbors::host_standard_dataset<uint8_t, int64_t>>*` | Default: `nullptr`. |

**Returns**

`void`

<a id="neighbors-cagra-serialize-to-hnswlib"></a>
### neighbors::cagra::serialize_to_hnswlib

Write the CAGRA built index as a base layer HNSW index to an output stream

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::device_padded_index<float>& index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<float>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Save a CAGRA build index in hnswlib base-layer-only serialized format

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::device_padded_index<float>& index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<float>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Write the CAGRA built index as a base layer HNSW index to an output stream

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::device_padded_index<half>& index,
std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<half>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Save a CAGRA build index in hnswlib base-layer-only serialized format

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::device_padded_index<half>& index,
std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<half>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Write the CAGRA built index as a base layer HNSW index to an output stream

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::device_padded_index<int8_t>& index,
std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<int8_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Save a CAGRA build index in hnswlib base-layer-only serialized format

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::device_padded_index<int8_t>& index,
std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<int8_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Write the CAGRA built index as a base layer HNSW index to an output stream

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::device_padded_index<uint8_t>& index,
std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<uint8_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Save a CAGRA build index in hnswlib base-layer-only serialized format

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::device_padded_index<uint8_t>& index,
std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::device_padded_index<uint8_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Write the CAGRA built index as a base layer HNSW index to an output stream.

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_padded_index<float>& index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Requires `dataset` — host builds do not store vectors in the index.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_padded_index<float>&` |  |
| `dataset` |  | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `neighbors::cagra::serialize_to_hnswlib`

Write the CAGRA built index as a base layer HNSW index to an output stream.

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::host_standard_index<float>& index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Requires `dataset` — host builds do not store vectors in the index.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | `const cuvs::neighbors::cagra::host_standard_index<float>&` |  |
| `dataset` |  | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | Default: `std::nullopt`. |

**Returns**

`void`

## CAGRA index build functions

<a id="neighbors-cagra-merge-algo"></a>
### neighbors::cagra::merge_algo

CAGRA index build functions

```cpp
enum class merge_algo {
  FASTENER,
  REBUILD
};
```

**Values**

| Name | Value |
| --- | --- |
| `FASTENER` | `` |
| `REBUILD` | `` |

<a id="neighbors-cagra-merge-params"></a>
### neighbors::cagra::merge_params

C++ controls for physical CAGRA index merge.

```cpp
struct merge_params {
  merge_algo algo;
  uint32_t levels;
  uint32_t root_fanout;
  uint32_t lower_fanout;
  double leader_fraction;
  uint32_t max_leaders;
  uint32_t leaf_size;
  uint32_t leaf_degree;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `algo` | [`merge_algo`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-merge-algo) |  |
| `levels` | `uint32_t` |  |
| `root_fanout` | `uint32_t` |  |
| `lower_fanout` | `uint32_t` |  |
| `leader_fraction` | `double` |  |
| `max_leaders` | `uint32_t` |  |
| `leaf_size` | `uint32_t` |  |
| `leaf_degree` | `uint32_t` |  |
