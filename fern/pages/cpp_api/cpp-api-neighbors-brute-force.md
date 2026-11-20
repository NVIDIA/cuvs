---
slug: api-reference/cpp-api-neighbors-brute-force
---

# Brute Force

_Source header: `cuvs/neighbors/brute_force.hpp`_

## Bruteforce index

<a id="neighbors-brute-force-index"></a>
### neighbors::brute_force::index

Brute Force index.

The index stores the dataset and norms for the dataset in device memory.

```cpp
template <typename T, typename DistT = T, typename DatasetViewT = void>
struct index;
```

<a id="neighbors-brute-force-index-index"></a>
### neighbors::brute_force::index::index

Construct an empty index.

```cpp
index(raft::resources const& handle);
```

Constructs an empty index. This index will either need to be trained with `build` or loaded from a saved copy with `deserialize`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::index::index`

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::host_matrix_view<const T, int64_t, raft::row_major> dataset_view,
std::optional<raft::device_vector<DistT, int64_t>>&& norms,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

Constructs a brute force index from a dataset. This lets us precompute norms for the dataset, providing a speed benefit over doing this at query time. This index will copy the host dataset onto the device, and take ownership of any precaculated norms.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::host_matrix_view<const T, int64_t, raft::row_major>` |  |
| `norms` |  | `std::optional<raft::device_vector<DistT, int64_t>>&&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::index::index`

Construct a brute force index and take ownership of a row-major device dataset.

```cpp
index(raft::resources const& res,
raft::device_matrix<T, int64_t, raft::row_major>&& dataset,
std::optional<raft::device_vector<DistT, int64_t>>&& norms,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::device_matrix<T, int64_t, raft::row_major>&&` |  |
| `norms` |  | `std::optional<raft::device_vector<DistT, int64_t>>&&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::index::index`

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
std::optional<raft::device_vector<DistT, int64_t>>&& norms,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

Constructs a brute force index from a dataset. This lets us precompute norms for the dataset, providing a speed benefit over doing this at query time. This index will store a non-owning reference to the dataset, but will move any norms supplied.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::device_matrix_view<const T, int64_t, raft::row_major>` |  |
| `norms` |  | `std::optional<raft::device_vector<DistT, int64_t>>&&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::index::index`

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

This class stores a non-owning reference to the dataset and norms. Having precomputed norms gives us a performance advantage at query time.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::device_matrix_view<const T, int64_t, raft::row_major>` |  |
| `norms_view` |  | `std::optional<raft::device_vector_view<const DistT, int64_t>>` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::index::index`

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
std::optional<raft::device_vector<DistT, int64_t>>&& norms,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

Constructs a brute force index from a dataset. This lets us precompute norms for the dataset, providing a speed benefit over doing this at query time. This index will store a non-owning reference to the dataset, but will move any norms supplied.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::device_matrix_view<const T, int64_t, raft::col_major>` |  |
| `norms` |  | `std::optional<raft::device_vector<DistT, int64_t>>&&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::index::index`

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

This class stores a non-owning reference to the dataset and norms, with the dataset being supplied on device in a col_major format

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::device_matrix_view<const T, int64_t, raft::col_major>` |  |
| `norms_view` |  | `std::optional<raft::device_vector_view<const DistT, int64_t>>` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

<a id="neighbors-brute-force-index-update-dataset"></a>
### neighbors::brute_force::index::update_dataset

Replace the dataset with a new dataset.

```cpp
void update_dataset(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::device_matrix_view<const T, int64_t, raft::row_major>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::index::update_dataset`

Replace the dataset with a new dataset.

```cpp
void update_dataset(raft::resources const& res,
raft::host_matrix_view<const T, int64_t, raft::row_major> dataset);
```

We create a copy of the dataset on the device. The index manages the lifetime of this copy.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::host_matrix_view<const T, int64_t, raft::row_major>` |  |

**Returns**

`void`

<a id="neighbors-brute-force-index-metric"></a>
### neighbors::brute_force::index::metric

Distance metric used for retrieval

```cpp
cuvs::distance::DistanceType metric() const noexcept;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="neighbors-brute-force-index-metric-arg"></a>
### neighbors::brute_force::index::metric_arg

Metric argument

```cpp
DistT metric_arg() const noexcept;
```

**Returns**

`DistT`

<a id="neighbors-brute-force-index-size"></a>
### neighbors::brute_force::index::size

Total length of the index (number of vectors).

```cpp
size_t size() const noexcept;
```

**Returns**

`size_t`

<a id="neighbors-brute-force-index-dim"></a>
### neighbors::brute_force::index::dim

Dimensionality of the data.

```cpp
size_t dim() const noexcept;
```

**Returns**

`size_t`

<a id="neighbors-brute-force-index-dataset"></a>
### neighbors::brute_force::index::dataset

Dataset [size, dim]

```cpp
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset() const noexcept;
```

**Returns**

`raft::device_matrix_view<const T, int64_t, raft::row_major>`

<a id="neighbors-brute-force-index-norms"></a>
### neighbors::brute_force::index::norms

Dataset norms

```cpp
raft::device_vector_view<const DistT, int64_t, raft::row_major> norms() const;
```

**Returns**

`raft::device_vector_view<const DistT, int64_t, raft::row_major>`

<a id="neighbors-brute-force-index-has-norms"></a>
### neighbors::brute_force::index::has_norms

Whether ot not this index has dataset norms

```cpp
inline bool has_norms() const noexcept;
```

**Returns**

`inline bool`

<a id="neighbors-brute-force-index"></a>
### neighbors::brute_force::index

SQ indexes borrow the codes and quantizer and own their distance statistics.

```cpp
template <typename T, typename DistT, typename DatasetViewT>
struct index;
```

**Additional overload:** `neighbors::brute_force::index::index`

Construct from an SQ view with the same requirements as build.

```cpp
index(raft::resources const& res, index_params const& params, DatasetViewT dataset);
```

The dataset's codes and quantizer must remain alive and unchanged while the index is used.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `index_params const&` |  |
| `dataset` |  | `DatasetViewT` |  |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::index::metric`

Distance metric used for retrieval.

```cpp
cuvs::distance::DistanceType metric() const noexcept;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

**Additional overload:** `neighbors::brute_force::index::metric_arg`

Metric argument.

```cpp
DistT metric_arg() const noexcept;
```

**Returns**

`DistT`

**Additional overload:** `neighbors::brute_force::index::size`

Number of corpus vectors.

```cpp
size_t size() const noexcept;
```

**Returns**

`size_t`

**Additional overload:** `neighbors::brute_force::index::dim`

Vector dimension.

```cpp
size_t dim() const noexcept;
```

**Returns**

`size_t`

**Additional overload:** `neighbors::brute_force::index::dataset`

Borrowed SQ dataset view, including its quantizer.

```cpp
DatasetViewT dataset() const noexcept;
```

**Returns**

`DatasetViewT`

**Additional overload:** `neighbors::brute_force::index::norms`

L2 squared code norms or normalized cosine norms; empty for inner product.

```cpp
raft::device_vector_view<const double, int64_t> norms() const;
```

**Returns**

`raft::device_vector_view<const double, int64_t>`

<a id="neighbors-brute-force-index-dot-corrections"></a>
### neighbors::brute_force::index::dot_corrections

Code sums for reconstructing inner products and cosine similarities, or empty for L2.

```cpp
raft::device_vector_view<const double, int64_t> dot_corrections() const;
```

**Returns**

`raft::device_vector_view<const double, int64_t>`

## Bruteforce index build

<a id="neighbors-brute-force-build"></a>
### neighbors::brute_force::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::brute_force::index<float, float>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::brute_force::index<float, float>`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index)

**Additional overload:** `neighbors::brute_force::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::brute_force::index<float, float>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host pointer to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::brute_force::index<float, float>`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index)

**Additional overload:** `neighbors::brute_force::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::brute_force::index<half, float>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::brute_force::index<half, float>`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index)

**Additional overload:** `neighbors::brute_force::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::brute_force::index<half, float>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host pointer to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::brute_force::index<half, float>`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index)

**Additional overload:** `neighbors::brute_force::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::col_major> dataset)
-> cuvs::neighbors::brute_force::index<float, float>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::col_major>` | a device pointer to a col-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::brute_force::index<float, float>`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index)

**Additional overload:** `neighbors::brute_force::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::col_major> dataset)
-> cuvs::neighbors::brute_force::index<half, float>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::col_major>` | a device pointer to a col-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::brute_force::index<half, float>`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index)

**Additional overload:** `neighbors::brute_force::build`

Build from a scalar-quantized device dataset.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& params,
cuvs::device_sq_dataset<float, int64_t, float>::const_view_type dataset)
-> cuvs::neighbors::brute_force::device_sq_index<float>;
```

The returned index borrows the codes and quantizer; the dataset must remain alive and unchanged. Supports global quantizers only. The dimension must be positive, divisible by four, and at most 131068. Supported metrics are InnerProduct, expanded and unexpanded L2 (including square-root variants), and CosineExpanded.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::brute_force::index_params&` | distance metric parameters |
| `dataset` | in | `cuvs::device_sq_dataset<float, int64_t, float>::const_view_type` | SQ dataset view |

**Returns**

`cuvs::neighbors::brute_force::device_sq_index<float>`

index accepting floating-point queries

**Additional overload:** `neighbors::brute_force::build`

Build an SQ index; see the float overload.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& params,
cuvs::device_sq_dataset<half, int64_t, half>::const_view_type dataset)
-> cuvs::neighbors::brute_force::device_sq_index<half>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | `const cuvs::neighbors::brute_force::index_params&` |  |
| `dataset` |  | `cuvs::device_sq_dataset<half, int64_t, half>::const_view_type` |  |

**Returns**

`cuvs::neighbors::brute_force::device_sq_index<half>`

**Additional overload:** `neighbors::brute_force::build`

Build an SQ index; see the float overload.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& params,
cuvs::device_sq_dataset<double, int64_t, double>::const_view_type dataset)
-> cuvs::neighbors::brute_force::device_sq_index<double>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | `const cuvs::neighbors::brute_force::index_params&` |  |
| `dataset` |  | `cuvs::device_sq_dataset<double, int64_t, double>::const_view_type` |  |

**Returns**

`cuvs::neighbors::brute_force::device_sq_index<double>`

## Bruteforce index search

<a id="neighbors-brute-force-search"></a>
### neighbors::brute_force::search

Search an SQ index by quantizing queries with the dataset's quantizer.

```cpp
void search(raft::resources const& handle,
const cuvs::neighbors::brute_force::search_params& params,
const cuvs::neighbors::brute_force::device_sq_index<float>& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
const cuvs::neighbors::filtering::base_filter& sample_filter =
cuvs::neighbors::filtering::none_sample_filter());
```

Uses INT8 GEMM with INT32 accumulation and returns float distances in the reconstructed vector space. Results are approximate relative to the original vectors. Use neighbors::refine with the original dataset for refinement. Prefiltering is not supported.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | `const cuvs::neighbors::brute_force::search_params&` | search parameters |
| `index` | in | `const cuvs::neighbors::brute_force::device_sq_index<float>&` | SQ brute-force index |
| `queries` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | row-major query vectors [n_queries, dim] |
| `neighbors` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | neighbor indices [n_queries, k], where 1 &lt;= k &lt;= index.size() |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | corresponding distances [n_queries, k] |
| `sample_filter` | in | `const cuvs::neighbors::filtering::base_filter&` | must be none_sample_filter<br />Default: `cuvs::neighbors::filtering::none_sample_filter()`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::search`

Search an SQ index with half queries; see the float overload.

```cpp
void search(raft::resources const& handle,
const cuvs::neighbors::brute_force::search_params& params,
const cuvs::neighbors::brute_force::device_sq_index<half>& index,
raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
const cuvs::neighbors::filtering::base_filter& sample_filter =
cuvs::neighbors::filtering::none_sample_filter());
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | `const cuvs::neighbors::brute_force::search_params&` |  |
| `index` |  | `const cuvs::neighbors::brute_force::device_sq_index<half>&` |  |
| `queries` |  | `raft::device_matrix_view<const half, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |
| `sample_filter` |  | `const cuvs::neighbors::filtering::base_filter&` | Default: `cuvs::neighbors::filtering::none_sample_filter()`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::search`

Search an SQ index with double queries; see the float overload.

```cpp
void search(raft::resources const& handle,
const cuvs::neighbors::brute_force::search_params& params,
const cuvs::neighbors::brute_force::device_sq_index<double>& index,
raft::device_matrix_view<const double, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
const cuvs::neighbors::filtering::base_filter& sample_filter =
cuvs::neighbors::filtering::none_sample_filter());
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | `const cuvs::neighbors::brute_force::search_params&` |  |
| `index` |  | `const cuvs::neighbors::brute_force::device_sq_index<double>&` |  |
| `queries` |  | `raft::device_matrix_view<const double, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |
| `sample_filter` |  | `const cuvs::neighbors::filtering::base_filter&` | Default: `cuvs::neighbors::filtering::none_sample_filter()`. |

**Returns**

`void`

## Sparse Brute Force index

<a id="neighbors-brute-force-sparse-index"></a>
### neighbors::brute_force::sparse_index

Sparse Brute Force index.

```cpp
template <typename T, typename IdxT>
struct sparse_index;
```

<a id="neighbors-brute-force-sparse-index-sparse-index"></a>
### neighbors::brute_force::sparse_index::sparse_index

Construct a sparse brute force sparse_index from dataset

```cpp
sparse_index(raft::resources const& res,
raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset,
cuvs::distance::DistanceType metric,
T metric_arg);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT>` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `metric_arg` |  | `T` |  |

**Returns**

`void`

<a id="neighbors-brute-force-sparse-index-metric"></a>
### neighbors::brute_force::sparse_index::metric

Distance metric used for retrieval

```cpp
cuvs::distance::DistanceType metric() const noexcept;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="neighbors-brute-force-sparse-index-metric-arg"></a>
### neighbors::brute_force::sparse_index::metric_arg

Metric argument

```cpp
T metric_arg() const noexcept;
```

**Returns**

`T`

## Sparse Brute Force index search

<a id="neighbors-brute-force-sparse-search-params"></a>
### neighbors::brute_force::sparse_search_params

Sparse Brute Force index search

```cpp
struct sparse_search_params;
```

## Bruteforce index serialize functions

<a id="neighbors-brute-force-serialize"></a>
### neighbors::brute_force::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::brute_force::index<half, float>& index,
bool include_dataset = true);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | data element type |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | [`const cuvs::neighbors::brute_force::index<half, float>&`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index) | brute force index |
| `include_dataset` | in | `bool` | whether to include the dataset in the serialized output<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::brute_force::index<float, float>& index,
bool include_dataset = true);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | data element type |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | [`const cuvs::neighbors::brute_force::index<float, float>&`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index) | brute force index |
| `include_dataset` | in | `bool` | whether to include the dataset in the serialized output<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::brute_force::index<half, float>& index,
bool include_dataset = true);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | [`const cuvs::neighbors::brute_force::index<half, float>&`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index) | brute force index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::brute_force::index<float, float>& index,
bool include_dataset = true);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | [`const cuvs::neighbors::brute_force::index<float, float>&`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index) | brute force index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file.<br />Default: `true`. |

**Returns**

`void`

<a id="neighbors-brute-force-deserialize"></a>
### neighbors::brute_force::deserialize

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::brute_force::index<half, float>* index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | [`cuvs::neighbors::brute_force::index<half, float>*`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index) | brute force index |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::brute_force::index<float, float>* index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | [`cuvs::neighbors::brute_force::index<float, float>*`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index) | brute force index |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::brute_force::index<half, float>* index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | [`cuvs::neighbors::brute_force::index<half, float>*`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index) | brute force index |

**Returns**

`void`

**Additional overload:** `neighbors::brute_force::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::brute_force::index<float, float>* index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | [`cuvs::neighbors::brute_force::index<float, float>*`](/api-reference/cpp-api-neighbors-brute-force#neighbors-brute-force-index) | brute force index |

**Returns**

`void`
