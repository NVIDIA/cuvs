---
slug: api-reference/cpp-api-neighbors-vamana
---

# Vamana

_Source header: `cuvs/neighbors/vamana.hpp`_

## Vamana index build parameters

<a id="neighbors-vamana-codebook-params"></a>
### neighbors::vamana::codebook_params

Parameters used to build quantized DiskANN index; to be generated using deserialize_codebooks()

```cpp
template <typename T = float>
struct codebook_params {
  int pq_codebook_size;
  int pq_dim;
  std::vector<T> pq_encoding_table;
  std::vector<T> rotation_matrix;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `pq_codebook_size` | `int` |  |
| `pq_dim` | `int` |  |
| `pq_encoding_table` | `std::vector<T>` |  |
| `rotation_matrix` | `std::vector<T>` |  |

<a id="neighbors-vamana-index-params"></a>
### neighbors::vamana::index_params

Parameters used to build DiskANN index

```cpp
struct index_params : cuvs::neighbors::index_params {
  uint32_t graph_degree;
  uint32_t visited_size;
  float vamana_iters;
  float alpha;
  float max_fraction;
  float batch_base;
  uint32_t queue_size;
  uint32_t reverse_batchsize;
  std::optional<codebook_params<float>> codebooks;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `graph_degree` | `uint32_t` | Maximum degree of graph; corresponds to the R parameter of Vamana algorithm in the literature. |
| `visited_size` | `uint32_t` | Maximum number of visited nodes per search during Vamana algorithm. Loosely corresponds to the L parameter in the literature. |
| `vamana_iters` | `float` | The number of times all vectors are inserted into the graph. If &gt; 1, all vectors are re-inserted to improve graph quality. |
| `alpha` | `float` | Used to determine how aggressive the pruning will be. |
| `max_fraction` | `float` | The maximum batch size is this fraction of the total dataset size. Larger gives faster build but lower graph quality. |
| `batch_base` | `float` | Base of growth rate of batch sizes * |
| `queue_size` | `uint32_t` | Size of candidate queue structure - should be (2^x)-1 |
| `reverse_batchsize` | `uint32_t` | Max batchsize of reverse edge processing (reduces memory footprint) |
| `codebooks` | [`std::optional<codebook_params<float>>`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-codebook-params) | Codebooks and related parameters |

## Vamana index type

<a id="neighbors-vamana-index"></a>
### neighbors::vamana::index

Vamana index.

The index stores the dataset and the Vamana graph in device memory.

```cpp
template <typename T, typename IdxT>
struct index;
```

<a id="neighbors-vamana-index-metric"></a>
### neighbors::vamana::index::metric

Distance metric used for clustering.

```cpp
[[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="neighbors-vamana-index-size"></a>
### neighbors::vamana::index::size

Total length of the index (number of vectors).

```cpp
[[nodiscard]] constexpr inline auto size() const noexcept -> IdxT;
```

**Returns**

`IdxT`

<a id="neighbors-vamana-index-dim"></a>
### neighbors::vamana::index::dim

Dimensionality of the data.

```cpp
[[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="neighbors-vamana-index-graph-degree"></a>
### neighbors::vamana::index::graph_degree

Graph degree

```cpp
[[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="neighbors-vamana-index-data"></a>
### neighbors::vamana::index::data

Dataset [size, dim]

```cpp
[[nodiscard]] inline auto data() const noexcept -> const cuvs::neighbors::dataset<int64_t>&;
```

**Returns**

[`const cuvs::neighbors::dataset<int64_t>&`](/api-reference/cpp-api-neighbors-common#neighbors-dataset)

<a id="neighbors-vamana-index-quantized-data"></a>
### neighbors::vamana::index::quantized_data

Quantized dataset [size, codes_rowlen]

```cpp
[[nodiscard]] inline auto quantized_data() const noexcept
-> raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>;
```

**Returns**

`raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>`

<a id="neighbors-vamana-index-graph"></a>
### neighbors::vamana::index::graph

vamana graph [size, graph-degree]

```cpp
[[nodiscard]] inline auto graph() const noexcept
-> raft::device_matrix_view<const IdxT, int64_t, raft::row_major>;
```

**Returns**

`raft::device_matrix_view<const IdxT, int64_t, raft::row_major>`

<a id="neighbors-vamana-index-medoid"></a>
### neighbors::vamana::index::medoid

Return the id of the vector selected as the medoid.

```cpp
[[nodiscard]] inline auto medoid() const noexcept -> IdxT;
```

**Returns**

`IdxT`

<a id="neighbors-vamana-index-index"></a>
### neighbors::vamana::index::index

```cpp
index(const index&)                    = delete;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | [`const index&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::vamana::index::index`

Construct an empty index.

```cpp
index(raft::resources const& res,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::vamana::index::index`

Construct an index from dataset and vamana graph

```cpp
template <typename data_accessor, typename graph_accessor>
index(raft::resources const& res,
cuvs::distance::DistanceType metric,
raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor> dataset,
raft::mdspan<const IdxT, raft::matrix_extent<int64_t>, raft::row_major, graph_accessor>
vamana_graph,
IdxT medoid_id);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `dataset` |  | `raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor>` |  |
| `vamana_graph` |  | `raft::mdspan<const IdxT, raft::matrix_extent<int64_t>, raft::row_major, graph_accessor>` |  |
| `medoid_id` |  | `IdxT` |  |

**Returns**

`void`

<a id="neighbors-vamana-index-update-graph"></a>
### neighbors::vamana::index::update_graph

Replace the graph with a new graph.

```cpp
void update_graph(raft::resources const& res,
raft::device_matrix_view<const IdxT, int64_t, raft::row_major> new_graph);
```

Since the new graph is a device array, we store a reference to that, and it is the caller's responsibility to ensure that knn_graph stays alive as long as the index.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `new_graph` |  | `raft::device_matrix_view<const IdxT, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::vamana::index::update_graph`

Replace the graph with a new graph.

```cpp
void update_graph(raft::resources const& res,
raft::host_matrix_view<const IdxT, int64_t, raft::row_major> new_graph);
```

We create a copy of the graph on the device. The index manages the lifetime of this copy.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `new_graph` |  | `raft::host_matrix_view<const IdxT, int64_t, raft::row_major>` |  |

**Returns**

`void`

<a id="neighbors-vamana-index-update-quantized-dataset"></a>
### neighbors::vamana::index::update_quantized_dataset

Replace the current quantized dataset with a new quantized dataset.

```cpp
void update_quantized_dataset(
raft::resources const& res,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_quantized_dataset);
```

We create a copy of the quantized dataset on the device. The index manages the lifetime of this copy.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `new_quantized_dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | the new quantized dataset for the index |

**Returns**

`void`

## Vamana index build functions

<a id="neighbors-vamana-build"></a>
### neighbors::vamana::build

Build the index from the dataset for efficient DiskANN search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::vamana::index_params& params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::vamana::index<float, uint32_t>;
```

The build utilities the Vamana insertion-based algorithm to create the graph. The algorithm starts with an empty graph and iteratively iserts batches of nodes. Each batch involves performing a greedy search for each vector to be inserted, and inserting it with edges to all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied to improve graph quality. The index_params struct controls the degree of the final graph.

The following distance metrics are supported:

- L2

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | [`const cuvs::neighbors::vamana::index_params&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index-params) | parameters for building the index |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a matrix view (device) to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::vamana::index<float, uint32_t>`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index)

**Additional overload:** `neighbors::vamana::build`

Build the index from the dataset for efficient DiskANN search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::vamana::index_params& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::vamana::index<float, uint32_t>;
```

The build utilities the Vamana insertion-based algorithm to create the graph. The algorithm starts with an empty graph and iteratively iserts batches of nodes. Each batch involves performing a greedy search for each vector to be inserted, and inserting it with edges to all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied to improve graph quality. The index_params struct controls the degree of the final graph.

The following distance metrics are supported:

- L2

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | [`const cuvs::neighbors::vamana::index_params&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index-params) | parameters for building the index |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a matrix view (host) to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::vamana::index<float, uint32_t>`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index)

**Additional overload:** `neighbors::vamana::build`

Build the index from the dataset for efficient DiskANN search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::vamana::index_params& params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::vamana::index<int8_t, uint32_t>;
```

The build utilities the Vamana insertion-based algorithm to create the graph. The algorithm starts with an empty graph and iteratively iserts batches of nodes. Each batch involves performing a greedy search for each vector to be inserted, and inserting it with edges to all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied to improve graph quality. The index_params struct controls the degree of the final graph.

The following distance metrics are supported:

- L2

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | [`const cuvs::neighbors::vamana::index_params&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index-params) | parameters for building the index |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a matrix view (device) to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::vamana::index<int8_t, uint32_t>`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index)

**Additional overload:** `neighbors::vamana::build`

Build the index from the dataset for efficient DiskANN search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::vamana::index_params& params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::vamana::index<int8_t, uint32_t>;
```

The build utilities the Vamana insertion-based algorithm to create the graph. The algorithm starts with an empty graph and iteratively iserts batches of nodes. Each batch involves performing a greedy search for each vector to be inserted, and inserting it with edges to all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied to improve graph quality. The index_params struct controls the degree of the final graph.

The following distance metrics are supported:

- L2

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | [`const cuvs::neighbors::vamana::index_params&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index-params) | parameters for building the index |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a matrix view (host) to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::vamana::index<int8_t, uint32_t>`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index)

**Additional overload:** `neighbors::vamana::build`

Build the index from the dataset for efficient DiskANN search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::vamana::index_params& params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::vamana::index<uint8_t, uint32_t>;
```

The build utilities the Vamana insertion-based algorithm to create the graph. The algorithm starts with an empty graph and iteratively iserts batches of nodes. Each batch involves performing a greedy search for each vector to be inserted, and inserting it with edges to all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied to improve graph quality. The index_params struct controls the degree of the final graph.

The following distance metrics are supported:

- L2

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | [`const cuvs::neighbors::vamana::index_params&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index-params) | parameters for building the index |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a matrix view (device) to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::vamana::index<uint8_t, uint32_t>`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index)

**Additional overload:** `neighbors::vamana::build`

Build the index from the dataset for efficient DiskANN search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::vamana::index_params& params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::vamana::index<uint8_t, uint32_t>;
```

The build utilities the Vamana insertion-based algorithm to create the graph. The algorithm starts with an empty graph and iteratively iserts batches of nodes. Each batch involves performing a greedy search for each vector to be inserted, and inserting it with edges to all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied to improve graph quality. The index_params struct controls the degree of the final graph.

The following distance metrics are supported:

- L2

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | [`const cuvs::neighbors::vamana::index_params&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index-params) | parameters for building the index |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a matrix view (host) to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::vamana::index<uint8_t, uint32_t>`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index)

## Vamana serialize functions

<a id="neighbors-vamana-serialize"></a>
### neighbors::vamana::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& file_prefix,
const cuvs::neighbors::vamana::index<float, uint32_t>& index,
bool include_dataset = true,
bool sector_aligned  = false);
```

Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `file_prefix` | in | `const std::string&` | prefix of path and name of index files |
| `index` | in | [`const cuvs::neighbors::vamana::index<float, uint32_t>&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index) | Vamana index |
| `include_dataset` | in | `bool` | whether or not to serialize the dataset<br />Default: `true`. |
| `sector_aligned` | in | `bool` | whether output file should be aligned to disk sectors of 4096 bytes<br />Default: `false`. |

**Returns**

`void`

**Additional overload:** `neighbors::vamana::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& file_prefix,
const cuvs::neighbors::vamana::index<int8_t, uint32_t>& index,
bool include_dataset = true,
bool sector_aligned  = false);
```

Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `file_prefix` | in | `const std::string&` | prefix of path and name of index files |
| `index` | in | [`const cuvs::neighbors::vamana::index<int8_t, uint32_t>&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index) | Vamana index |
| `include_dataset` | in | `bool` | whether or not to serialize the dataset<br />Default: `true`. |
| `sector_aligned` | in | `bool` | whether output file should be aligned to disk sectors of 4096 bytes<br />Default: `false`. |

**Returns**

`void`

**Additional overload:** `neighbors::vamana::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& file_prefix,
const cuvs::neighbors::vamana::index<uint8_t, uint32_t>& index,
bool include_dataset = true,
bool sector_aligned  = false);
```

Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `file_prefix` | in | `const std::string&` | prefix of path and name of index files |
| `index` | in | [`const cuvs::neighbors::vamana::index<uint8_t, uint32_t>&`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-index) | Vamana index |
| `include_dataset` | in | `bool` | whether or not to serialize the dataset<br />Default: `true`. |
| `sector_aligned` | in | `bool` | whether output file should be aligned to disk sectors of 4096 bytes<br />Default: `false`. |

**Returns**

`void`

## Vamana codebook functions

<a id="neighbors-vamana-deserialize-codebooks"></a>
### neighbors::vamana::deserialize_codebooks

Construct codebook parameters from input codebook files

```cpp
auto deserialize_codebooks(const std::string& codebook_prefix, const int dim)
-> codebook_params<float>;
```

Expects pq pivots file at "$\{codebook_prefix\}_pq_pivots.bin" and rotation matrix file at "$\{codebook_prefix\}_pq_pivots.bin_rotation_matrix.bin".

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `codebook_prefix` | in | `const std::string&` | path prefix to pq pivots and rotation matrix files |
| `dim` | in | `const int` | dimension of vectors in dataset |

**Returns**

[`codebook_params<float>`](/api-reference/cpp-api-neighbors-vamana#neighbors-vamana-codebook-params)
