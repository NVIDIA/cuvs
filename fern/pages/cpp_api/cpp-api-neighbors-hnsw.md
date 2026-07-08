---
slug: api-reference/cpp-api-neighbors-hnsw
---

# HNSW

_Source header: `cuvs/neighbors/hnsw.hpp`_

## hnswlib index wrapper params

<a id="neighbors-hnsw-hnswhierarchy"></a>
### neighbors::hnsw::HnswHierarchy

Hierarchy for HNSW index when converting from CAGRA index

NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only index.

```cpp
enum class HnswHierarchy {
  NONE
};
```

**Values**

| Name | Value |
| --- | --- |
| `NONE` | `` |

<a id="neighbors-hnsw-deprecated"></a>
### neighbors::hnsw::[[deprecated

Create a CAGRA index parameters compatible with HNSW index

```cpp
[[deprecated("Use cagra::index_params::from_hnsw_params instead")]]
cuvs::neighbors::cagra::index_params to_cagra_params(
raft::matrix_extent<int64_t> dataset,
int M,
int ef_construction,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

* IMPORTANT NOTE *

The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce the same recalls and QPS for the same parameter `ef`. The graphs are different internally. For the same `ef`, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS. However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the curve).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `"Use cagra::index_params::from_hnsw_params instead"` |  |

**Returns**

`void`

## hnswlib index wrapper

<a id="neighbors-hnsw-index"></a>
### neighbors::hnsw::index

hnswlib index wrapper

```cpp
template <typename T>
struct index;
```

<a id="neighbors-hnsw-index-index"></a>
### neighbors::hnsw::index::index

load a base-layer-only hnswlib index originally saved from a built CAGRA index. This is a virtual class and it cannot be used directly. To create an index, use the factory function `cuvs::neighbors::hnsw::from_cagra` from the header `cuvs/neighbors/hnsw.hpp`

```cpp
index(int dim, cuvs::distance::DistanceType metric, HnswHierarchy hierarchy = HnswHierarchy::NONE);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `hierarchy` | in | [`HnswHierarchy`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-hnswhierarchy) | hierarchy used for upper HNSW layers<br />Default: `HnswHierarchy::NONE`. |

**Returns**

`void`

<a id="neighbors-hnsw-index-get-index"></a>
### neighbors::hnsw::index::get_index

Get underlying index

```cpp
virtual void const* get_index() const = 0;
```

**Returns**

`virtual void const*`

<a id="neighbors-hnsw-index-set-ef"></a>
### neighbors::hnsw::index::set_ef

Set ef for search

```cpp
virtual void set_ef(int ef) const = 0;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `ef` |  | `int` |  |

**Returns**

`virtual void`

<a id="neighbors-hnsw-index-file-path"></a>
### neighbors::hnsw::index::file_path

Get file path for disk-backed index

```cpp
virtual std::string file_path() const;
```

**Returns**

`virtual std::string`

## HNSW index extend parameters

<a id="neighbors-hnsw-extend-params"></a>
### neighbors::hnsw::extend_params

HNSW index extend parameters

```cpp
struct extend_params {
  int num_threads;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `num_threads` | `int` | Number of host threads to use to add additional vectors to the index. Value of 0 automatically maximizes parallelism. |

## Build HNSW index on the GPU

<a id="neighbors-hnsw-build"></a>
### neighbors::hnsw::build

Build an HNSW index on the GPU

```cpp
std::unique_ptr<index<float>> build(
raft::resources const& res,
const index_params& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset);
```

The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph built by the HNSW.

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality. Additional GPU graph build settings are selected automatically. Set `params.graph_build_params` only when partitioned or disk-backed ACE construction needs explicit configuration.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters with optional ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<float>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::build`

Build an HNSW index on the GPU

```cpp
std::unique_ptr<index<half>> build(
raft::resources const& res,
const index_params& params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset);
```

The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph built by the HNSW.

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality. Additional GPU graph build settings are selected automatically. Set `params.graph_build_params` only when partitioned or disk-backed ACE construction needs explicit configuration.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters with optional ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<half>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::build`

Build an HNSW index on the GPU

```cpp
std::unique_ptr<index<uint8_t>> build(
raft::resources const& res,
const index_params& params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset);
```

The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph built by the HNSW.

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality. Additional GPU graph build settings are selected automatically. Set `params.graph_build_params` only when partitioned or disk-backed ACE construction needs explicit configuration.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters with optional ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<uint8_t>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::build`

Build an HNSW index on the GPU

```cpp
std::unique_ptr<index<int8_t>> build(
raft::resources const& res,
const index_params& params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset);
```

The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph built by the HNSW.

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality. Additional GPU graph build settings are selected automatically. Set `params.graph_build_params` only when partitioned or disk-backed ACE construction needs explicit configuration.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters with optional ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<int8_t>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

## Load CAGRA index as hnswlib index

<a id="neighbors-hnsw-from-cagra"></a>
### neighbors::hnsw::from_cagra

Construct an hnswlib index from a CAGRA index NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU` or `GPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library. With `GPU`, the hierarchy is built on the GPU.

```cpp
std::unique_ptr<index<float>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::index<float, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `cagra_index` | in | [`const cuvs::neighbors::cagra::index<float, uint32_t>&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-index) | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU` or `GPU`<br /><br />Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<float>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU` or `GPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library. With `GPU`, the hierarchy is built on the GPU.

```cpp
std::unique_ptr<index<half>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::index<half, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `cagra_index` | in | [`const cuvs::neighbors::cagra::index<half, uint32_t>&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-index) | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU` or `GPU`<br /><br />Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<half>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU` or `GPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library. With `GPU`, the hierarchy is built on the GPU.

```cpp
std::unique_ptr<index<uint8_t>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `cagra_index` | in | [`const cuvs::neighbors::cagra::index<uint8_t, uint32_t>&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-index) | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU` or `GPU`<br /><br />Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<uint8_t>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU` or `GPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library. With `GPU`, the hierarchy is built on the GPU.

```cpp
std::unique_ptr<index<int8_t>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::index<int8_t, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `cagra_index` | in | [`const cuvs::neighbors::cagra::index<int8_t, uint32_t>&`](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-index) | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU` or `GPU`<br /><br />Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<int8_t>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

## Extend HNSW index with additional vectors

<a id="neighbors-hnsw-extend"></a>
### neighbors::hnsw::extend

Add new vectors to an HNSW index NOTE: A base-layer-only index with hierarchy `NONE` cannot be extended. Indexes with hierarchy `CPU` or `GPU` can be extended.

```cpp
void extend(raft::resources const& res,
const extend_params& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
index<float>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const extend_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-extend-params) | configure the extend |
| `additional_dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, index-&gt;dim()] |
| `idx` | inout | [`index<float>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index to extend |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::extend`

Add new vectors to an HNSW index NOTE: A base-layer-only index with hierarchy `NONE` cannot be extended. Indexes with hierarchy `CPU` or `GPU` can be extended.

```cpp
void extend(raft::resources const& res,
const extend_params& params,
raft::host_matrix_view<const half, int64_t, raft::row_major> additional_dataset,
index<half>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const extend_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-extend-params) | configure the extend |
| `additional_dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, index-&gt;dim()] |
| `idx` | inout | [`index<half>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index to extend |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::extend`

Add new vectors to an HNSW index NOTE: A base-layer-only index with hierarchy `NONE` cannot be extended. Indexes with hierarchy `CPU` or `GPU` can be extended.

```cpp
void extend(raft::resources const& res,
const extend_params& params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
index<uint8_t>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const extend_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-extend-params) | configure the extend |
| `additional_dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, index-&gt;dim()] |
| `idx` | inout | [`index<uint8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index to extend |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::extend`

Add new vectors to an HNSW index NOTE: A base-layer-only index with hierarchy `NONE` cannot be extended. Indexes with hierarchy `CPU` or `GPU` can be extended.

```cpp
void extend(raft::resources const& res,
const extend_params& params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
index<int8_t>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const extend_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-extend-params) | configure the extend |
| `additional_dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, index-&gt;dim()] |
| `idx` | inout | [`index<int8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index to extend |

**Returns**

`void`

## Build CAGRA index and search with hnswlib

<a id="neighbors-hnsw-search-params"></a>
### neighbors::hnsw::search_params

Build CAGRA index and search with hnswlib

```cpp
struct search_params : cuvs::neighbors::search_params {
  int ef;
  int num_threads;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `ef` | `int` |  |
| `num_threads` | `int` |  |

## Search hnswlib index

<a id="neighbors-hnsw-search"></a>
### neighbors::hnsw::search

Search HNSW index constructed from a CAGRA index NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib.

```cpp
void search(raft::resources const& res,
const search_params& params,
const index<float>& idx,
raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const search_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-search-params) | configure the search |
| `idx` | in | [`const index<float>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | cagra index |
| `queries` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::search`

Search HNSW index constructed from a CAGRA index NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib.

```cpp
void search(raft::resources const& res,
const search_params& params,
const index<half>& idx,
raft::host_matrix_view<const half, int64_t, raft::row_major> queries,
raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const search_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-search-params) | configure the search |
| `idx` | in | [`const index<half>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | cagra index |
| `queries` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::search`

Search HNSWindex constructed from a CAGRA index NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib.

```cpp
void search(raft::resources const& res,
const search_params& params,
const index<uint8_t>& idx,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const search_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-search-params) | configure the search |
| `idx` | in | [`const index<uint8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | cagra index |
| `queries` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::search`

Search HNSW index constructed from a CAGRA index NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib.

```cpp
void search(raft::resources const& res,
const search_params& params,
const index<int8_t>& idx,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> queries,
raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const search_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-search-params) | configure the search |
| `idx` | in | [`const index<int8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | cagra index |
| `queries` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

## Serialize and deserialize HNSW indexes

<a id="neighbors-hnsw-serialize"></a>
### neighbors::hnsw::serialize

Serialize the HNSW index to file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, with hierarchy `CPU` or `GPU`, the saved index is compatible with the original hnswlib library.

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<float>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the file to save the serialized HNSW index |
| `idx` | in | [`const index<float>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::serialize`

Serialize the HNSW index to file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, with hierarchy `CPU` or `GPU`, the saved index is compatible with the original hnswlib library.

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<half>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the file to save the serialized HNSW index |
| `idx` | in | [`const index<half>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::serialize`

Serialize the HNSW index to file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, with hierarchy `CPU` or `GPU`, the saved index is compatible with the original hnswlib library.

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<uint8_t>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the file to save the serialized HNSW index |
| `idx` | in | [`const index<uint8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::serialize`

Serialize the HNSW index to file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, with hierarchy `CPU` or `GPU`, the saved index is compatible with the original hnswlib library.

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<int8_t>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the file to save the serialized HNSW index |
| `idx` | in | [`const index<int8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |

**Returns**

`void`

<a id="neighbors-hnsw-deserialize"></a>
### neighbors::hnsw::deserialize

Deserialize an HNSW index from a file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, with hierarchy `CPU` or `GPU`, the saved index is compatible with the original hnswlib library. NOTE: `params.hierarchy` must match the hierarchy the index was serialized with.

```cpp
void deserialize(raft::resources const& res,
const index_params& params,
const std::string& filename,
int dim,
cuvs::distance::DistanceType metric,
index<float>** index);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the file containing the serialized HNSW index |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct"); takes precedence over `params.metric` |
| `index` | out | [`index<float>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

Deserialize an HNSW index from a file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, with hierarchy `CPU` or `GPU`, the saved index is compatible with the original hnswlib library. NOTE: `params.hierarchy` must match the hierarchy the index was serialized with.

```cpp
void deserialize(raft::resources const& res,
const index_params& params,
const std::string& filename,
int dim,
cuvs::distance::DistanceType metric,
index<half>** index);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the file containing the serialized HNSW index |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct"); takes precedence over `params.metric` |
| `index` | out | [`index<half>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

Deserialize an HNSW index from a file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, with hierarchy `CPU` or `GPU`, the saved index is compatible with the original hnswlib library. NOTE: `params.hierarchy` must match the hierarchy the index was serialized with.

```cpp
void deserialize(raft::resources const& res,
const index_params& params,
const std::string& filename,
int dim,
cuvs::distance::DistanceType metric,
index<uint8_t>** index);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the file containing the serialized HNSW index |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct"); takes precedence over `params.metric` |
| `index` | out | [`index<uint8_t>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

Deserialize an HNSW index from a file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, with hierarchy `CPU` or `GPU`, the saved index is compatible with the original hnswlib library. NOTE: `params.hierarchy` must match the hierarchy the index was serialized with.

```cpp
void deserialize(raft::resources const& res,
const index_params& params,
const std::string& filename,
int dim,
cuvs::distance::DistanceType metric,
index<int8_t>** index);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the file containing the serialized HNSW index |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct"); takes precedence over `params.metric` |
| `index` | out | [`index<int8_t>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | hnsw index |

**Returns**

`void`
