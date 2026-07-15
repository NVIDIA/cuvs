---
slug: api-reference/cpp-api-neighbors-nn-descent
---

# NN Descent

_Source header: `cuvs/neighbors/nn_descent.hpp`_

## The nn-descent algorithm parameters.

<a id="neighbors-nn-descent-dist-comp-dtype"></a>
### neighbors::nn_descent::DIST_COMP_DTYPE

Dtype to use for distance computation

```cpp
enum class DIST_COMP_DTYPE {
  AUTO = 0,
  FP32 = 1,
  FP16 = 2
};
```

**Values**

| Name | Value | Description |
| --- | --- | --- |
| `AUTO` | `0` | Automatically determine the best dtype for distance computation based on the dataset dimensions. |
| `FP32` | `1` | Use fp32 distance computation for better precision at the cost of performance and memory usage. |
| `FP16` | `2` | Use fp16 distance computation. |

<a id="neighbors-nn-descent-index-params"></a>
### neighbors::nn_descent::index_params

Parameters used to build an nn-descent index

```cpp
struct index_params : cuvs::neighbors::index_params {
  size_t graph_degree;
  size_t intermediate_graph_degree;
  size_t max_iterations;
  float termination_threshold;
  bool return_distances;
  DIST_COMP_DTYPE dist_comp_dtype;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `graph_degree` | `size_t` | For an input dataset of dimensions (N, D), determines the final dimensions of the all-neighbors knn graph which turns out to be of dimensions (N, graph_degree) |
| `intermediate_graph_degree` | `size_t` | Internally, nn-descent builds an all-neighbors knn graph of dimensions (N, intermediate_graph_degree) before selecting the final `graph_degree` neighbors. It's recommended that `intermediate_graph_degree` &gt;= 1.5 * graph_degree |
| `max_iterations` | `size_t` | The number of iterations that nn-descent will refine the graph for. More iterations produce a better quality graph at cost of performance |
| `termination_threshold` | `float` | The delta at which nn-descent will terminate its iterations |
| `return_distances` | `bool` | Boolean to decide whether to return distances array |
| `dist_comp_dtype` | [`DIST_COMP_DTYPE`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-dist-comp-dtype) | dtype to use for distance computation.<br />Defaults to `AUTO` which automatically determines the best dtype for distance computation based on the dataset dimensions.<br />Use `FP32` for better precision at the cost of performance and memory usage. This option is only valid when data type is fp32.<br />Use `FP16` for better performance and memory usage at the cost of precision. |

<a id="neighbors-nn-descent-index-params-index-params"></a>
### neighbors::nn_descent::index_params::index_params

Construct NN descent parameters for a specific kNN graph degree

```cpp
index_params(size_t graph_degree                 = 64,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `graph_degree` |  | `size_t` | output graph degree<br />Default: `64`. |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use<br />Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`void`

## nn-descent index

<a id="neighbors-nn-descent-index"></a>
### neighbors::nn_descent::index

nn-descent Build an nn-descent index The index contains an all-neighbors graph of the input dataset stored in host memory of dimensions (n_rows, n_cols)

Reference: Hui Wang, Wan-Lei Zhao, Xiangxiang Zeng, and Jianye Yang. 2021. Fast k-NN Graph Construction by GPU based NN-Descent. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM '21). Association for Computing Machinery, New York, NY, USA, 1929–1938. https://doi.org/10.1145/3459637.3482344

```cpp
template <typename IdxT>
struct index;
```

<a id="neighbors-nn-descent-index-index"></a>
### neighbors::nn_descent::index::index

Construct a new index object

```cpp
index(raft::resources const& res,
int64_t n_rows,
int64_t n_cols,
bool return_distances               = false,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

This constructor creates an nn-descent index which is a knn-graph in host memory. The type of the knn-graph is a dense raft::host_matrix and dimensions are (n_rows, n_cols).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` | raft::resources is an object managing resources |
| `n_rows` |  | `int64_t` | number of rows in knn-graph |
| `n_cols` |  | `int64_t` | number of cols in knn-graph |
| `return_distances` |  | `bool` | whether to return distances<br />Default: `false`. |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use<br />Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`void`

**Additional overload:** `neighbors::nn_descent::index::index`

Construct a new index object

```cpp
index(raft::resources const& res,
raft::host_matrix_view<IdxT, int64_t, raft::row_major> graph_view,
std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances_view =
std::nullopt,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

This constructor creates an nn-descent index using a user allocated host memory knn-graph. The type of the knn-graph is a dense raft::host_matrix and dimensions are (n_rows, n_cols).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` | raft::resources is an object managing resources |
| `graph_view` |  | `raft::host_matrix_view<IdxT, int64_t, raft::row_major>` | raft::host_matrix_view&lt;IdxT, int64_t, raft::row_major&gt; for storing knn-graph |
| `distances_view` |  | `std::optional<raft::device_matrix_view<float, int64_t, row_major>>` | optional raft::device_matrix_view&lt;float, int64_t, row_major&gt; for storing distances<br />Default: `std::nullopt`. |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use<br />Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`void`

<a id="neighbors-nn-descent-index-metric"></a>
### neighbors::nn_descent::index::metric

Distance metric used for clustering.

```cpp
[[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="neighbors-nn-descent-index-size"></a>
### neighbors::nn_descent::index::size

Total length of the index (number of vectors).

```cpp
[[nodiscard]] constexpr inline auto size() const noexcept -> IdxT;
```

**Returns**

`IdxT`

<a id="neighbors-nn-descent-index-graph-degree"></a>
### neighbors::nn_descent::index::graph_degree

Graph degree

```cpp
[[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

<a id="neighbors-nn-descent-index-graph"></a>
### neighbors::nn_descent::index::graph

neighborhood graph [size, graph-degree]

```cpp
[[nodiscard]] inline auto graph() noexcept
-> raft::host_matrix_view<IdxT, int64_t, raft::row_major>;
```

**Returns**

`raft::host_matrix_view<IdxT, int64_t, raft::row_major>`

<a id="neighbors-nn-descent-index-distances"></a>
### neighbors::nn_descent::index::distances

neighborhood graph distances [size, graph-degree]

```cpp
[[nodiscard]] inline auto distances() noexcept
-> std::optional<device_matrix_view<float, int64_t, row_major>>;
```

**Returns**

`std::optional<device_matrix_view<float, int64_t, row_major>>`

## nn-descent index build

<a id="neighbors-nn-descent-build"></a>
### neighbors::nn_descent::build

Build nn-descent Index with dataset in device memory

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;
```

The following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- CosineExpanded
- InnerProduct
- L1

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft::resources is an object managing resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index-params) | an instance of nn_descent::index_params that are parameters to run the nn-descent algorithm |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | raft::device_matrix_view input dataset expected to be located in device memory |
| `graph` | in | `std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>>` | optional raft::host_matrix_view&lt;uint32_t, int64_t, raft::row_major&gt; for owning the output graph<br />Default: `std::nullopt`. |

**Returns**

[`cuvs::neighbors::nn_descent::index<uint32_t>`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index)

**Additional overload:** `neighbors::nn_descent::build`

Build nn-descent Index with dataset in host memory

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;
```

The following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- CosineExpanded
- InnerProduct
- L1

Usage example:

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | data-type of the input dataset |
| `IdxT` | `` | data-type for the output index |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` | raft::resources is an object managing resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index-params) | an instance of nn_descent::index_params that are parameters to run the nn-descent algorithm |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | raft::host_matrix_view input dataset expected to be located in host memory |
| `graph` | in | `std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>>` | optional raft::host_matrix_view&lt;uint32_t, int64_t, raft::row_major&gt; for owning the output graph<br />Default: `std::nullopt`. |

**Returns**

[`cuvs::neighbors::nn_descent::index<uint32_t>`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index)

**Additional overload:** `neighbors::nn_descent::build`

Build nn-descent Index with dataset in device memory

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;
```

The following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- CosineExpanded
- InnerProduct
- L1

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft::resources is an object managing resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index-params) | an instance of nn_descent::index_params that are parameters to run the nn-descent algorithm |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | raft::device_matrix_view input dataset expected to be located in device memory |
| `graph` | in | `std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>>` | optional raft::host_matrix_view&lt;uint32_t, int64_t, raft::row_major&gt; for owning the output graph<br />Default: `std::nullopt`. |

**Returns**

[`cuvs::neighbors::nn_descent::index<uint32_t>`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index)

**Additional overload:** `neighbors::nn_descent::build`

Build nn-descent Index with dataset in host memory

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;
```

The following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- CosineExpanded
- InnerProduct
- L1

Usage example:

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | data-type of the input dataset |
| `IdxT` | `` | data-type for the output index |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` | raft::resources is an object managing resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index-params) | an instance of nn_descent::index_params that are parameters to run the nn-descent algorithm |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | raft::host_matrix_view input dataset expected to be located in host memory |
| `graph` | in | `std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>>` | optional raft::host_matrix_view&lt;uint32_t, int64_t, raft::row_major&gt; for owning the output graph<br />Default: `std::nullopt`. |

**Returns**

[`cuvs::neighbors::nn_descent::index<uint32_t>`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index)

**Additional overload:** `neighbors::nn_descent::build`

Build nn-descent Index with dataset in device memory

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;
```

The following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- CosineExpanded
- InnerProduct
- L1
- BitwiseHamming

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft::resources is an object managing resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index-params) | an instance of nn_descent::index_params that are parameters to run the nn-descent algorithm |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::device_matrix_view input dataset expected to be located in device memory |
| `graph` | in | `std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>>` | optional raft::host_matrix_view&lt;uint32_t, int64_t, raft::row_major&gt; for owning the output graph<br />Default: `std::nullopt`. |

**Returns**

[`cuvs::neighbors::nn_descent::index<uint32_t>`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index)

**Additional overload:** `neighbors::nn_descent::build`

Build nn-descent Index with dataset in host memory

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;
```

The following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- CosineExpanded
- InnerProduct
- L1
- BitwiseHamming

Usage example:

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | data-type of the input dataset |
| `IdxT` | `` | data-type for the output index |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` | raft::resources is an object managing resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index-params) | an instance of nn_descent::index_params that are parameters to run the nn-descent algorithm |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::host_matrix_view input dataset expected to be located in host memory |
| `graph` | in | `std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>>` | optional raft::host_matrix_view&lt;uint32_t, int64_t, raft::row_major&gt; for owning the output graph<br />Default: `std::nullopt`. |

**Returns**

[`cuvs::neighbors::nn_descent::index<uint32_t>`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index)

**Additional overload:** `neighbors::nn_descent::build`

Build nn-descent Index with dataset in device memory

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;
```

The following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- CosineExpanded
- InnerProduct
- L1
- BitwiseHamming

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft::resources is an object managing resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index-params) | an instance of nn_descent::index_params that are parameters to run the nn-descent algorithm |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::device_matrix_view input dataset expected to be located in device memory |
| `graph` | in | `std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>>` | optional raft::host_matrix_view&lt;uint32_t, int64_t, raft::row_major&gt; for owning the output graph<br />Default: `std::nullopt`. |

**Returns**

[`cuvs::neighbors::nn_descent::index<uint32_t>`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index)

**Additional overload:** `neighbors::nn_descent::build`

Build nn-descent Index with dataset in host memory

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;
```

The following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- CosineExpanded
- InnerProduct
- L1
- BitwiseHamming

Usage example:

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | data-type of the input dataset |
| `IdxT` | `` | data-type for the output index |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` | raft::resources is an object managing resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index-params) | an instance of nn_descent::index_params that are parameters to run the nn-descent algorithm |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::host_matrix_view input dataset expected to be located in host memory |
| `graph` | in | `std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>>` | optional raft::host_matrix_view&lt;uint32_t, int64_t, raft::row_major&gt; for owning the output graph<br />Default: `std::nullopt`. |

**Returns**

[`cuvs::neighbors::nn_descent::index<uint32_t>`](/api-reference/cpp-api-neighbors-nn-descent#neighbors-nn-descent-index)
