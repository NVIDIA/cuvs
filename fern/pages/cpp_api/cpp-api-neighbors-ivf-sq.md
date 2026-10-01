---
slug: api-reference/cpp-api-neighbors-ivf-sq
---

# IVF SQ

_Source header: `cuvs/neighbors/ivf_sq.hpp`_

## IVF-SQ index build parameters

<a id="kindexgroupsize"></a>
### kIndexGroupSize

IVF-SQ index build parameters

```cpp
constexpr static uint32_t kIndexGroupSize = 32;
```

<a id="neighbors-ivf-sq-index-params"></a>
### neighbors::ivf_sq::index_params

IVF-SQ index build parameters.

IVF-SQ currently uses 8-bit scalar quantization, storing one `uint8_t` code per vector dimension.

```cpp
struct index_params : cuvs::neighbors::index_params {
  uint32_t n_lists;
  uint32_t kmeans_n_iters;
  uint32_t max_train_points_per_cluster;
  bool conservative_memory_allocation;
  bool add_data_on_build;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_lists` | `uint32_t` | The number of inverted lists (clusters) |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (index building). |
| `max_train_points_per_cluster` | `uint32_t` | The number of data vectors (per cluster) to use during iterative kmeans building. |
| `conservative_memory_allocation` | `bool` | By default, the algorithm allocates more space than necessary for individual clusters (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of data copies during repeated calls to `extend` (extending the database).<br /><br />The alternative is the conservative allocation behavior; when enabled, the algorithm always allocates the minimum amount of memory required to store the given number of records. Set this flag to `true` if you prefer to use as little GPU memory for the database as possible. |
| `add_data_on_build` | `bool` | Whether to add the dataset content to the index, i.e.:<br /><br />- `true` means the index is filled with the dataset vectors and ready to search after calling `build`.<br />- `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but the index is left empty; you'd need to call `extend` on the index afterwards to populate it. |

## IVF-SQ index search parameters

<a id="neighbors-ivf-sq-search-params"></a>
### neighbors::ivf_sq::search_params

IVF-SQ index search parameters

```cpp
struct search_params : cuvs::neighbors::search_params {
  uint32_t n_probes;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |

## IVF-SQ list storage spec

<a id="neighbors-ivf-sq-list-spec"></a>
### neighbors::ivf_sq::list_spec

IVF-SQ list storage spec

```cpp
template <typename SizeT, typename CodeT, typename IdxT>
struct list_spec {
  SizeT align_max;
  SizeT align_min;
  uint32_t dim;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `align_max` | `SizeT` |  |
| `align_min` | `SizeT` |  |
| `dim` | `uint32_t` |  |

## IVF-SQ index

<a id="neighbors-ivf-sq-index"></a>
### neighbors::ivf_sq::index

IVF-SQ index.

In the IVF-SQ index, a database vector is first assigned to the nearest cluster center using an inverted file (IVF) structure, and then compressed using scalar quantization (SQ).

Scalar quantization independently maps each dimension of the per-cluster residual (the input vector minus its assigned centroid) to a fixed-width integer code. For 8-bit quantization (`uint8_t`), each residual component is linearly mapped to an integer in [0, 255] using learned per-dimension minimum (`sq_vmin`) and step-size (`sq_delta`) values.

For a vector component `x_i`, centroid component `centroid_i`, residual minimum `vmin_i`, and quantization step `delta_i`, the stored code is:

$$code_i = clamp(round((x_i - centroid_i - vmin_i) / delta_i), 0, 255)$$

The corresponding reconstructed component is:

$$x_i \approx centroid_i + vmin_i + code_i \cdot delta_i$$

This provides a compact representation (1 byte per dimension) while preserving the relative distances between vectors with high fidelity, offering a good trade-off between index size, search speed, and recall compared to flat (uncompressed) and product-quantized (PQ) representations.

Note: `CodeT` is the storage type for scalar-quantized residual codes in the inverted lists, not the input dataset type. The public build and search APIs accept `float` and `half` input vectors, then store the quantized residual components as `CodeT` inside the index. Currently, IVF-SQ supports only `uint8_t` codes, so use `CodeT = uint8_t`. Each code uses the full 8-bit range [0, 255].

```cpp
template <typename CodeT>
struct index;
```

## IVF-SQ index build

<a id="neighbors-ivf-sq-build"></a>
### neighbors::ivf_sq::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_sq::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_sq::index<uint8_t>;
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2SqrtExpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | [`const cuvs::neighbors::ivf_sq::index_params&`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index-params) | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_sq::index<uint8_t>`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index)

**Additional overload:** `neighbors::ivf_sq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_sq::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_sq::index<uint8_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | [`const cuvs::neighbors::ivf_sq::index_params&`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index-params) | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_sq::index<uint8_t>`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index)

**Additional overload:** `neighbors::ivf_sq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_sq::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_sq::index<uint8_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | [`const cuvs::neighbors::ivf_sq::index_params&`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index-params) | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host pointer to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_sq::index<uint8_t>`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index)

**Additional overload:** `neighbors::ivf_sq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_sq::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_sq::index<uint8_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | [`const cuvs::neighbors::ivf_sq::index_params&`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index-params) | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host pointer to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_sq::index<uint8_t>`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index)

## IVF-SQ index extend

<a id="neighbors-ivf-sq-extend"></a>
### neighbors::ivf_sq::extend

Extend the index with the new data in-place.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_sq::index<uint8_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_sq::index<uint8_t>*`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index) | pointer to ivf_sq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_sq::extend`

Extend the index with the new data in-place.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_sq::index<uint8_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_sq::index<uint8_t>*`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index) | pointer to ivf_sq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_sq::extend`

Extend the index with the new data in-place.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_sq::index<uint8_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_sq::index<uint8_t>*`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index) | pointer to ivf_sq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_sq::extend`

Extend the index with the new data in-place.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_sq::index<uint8_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_sq::index<uint8_t>*`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index) | pointer to ivf_sq::index |

**Returns**

`void`

## IVF-SQ index serialize

<a id="neighbors-ivf-sq-serialize"></a>
### neighbors::ivf_sq::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::ivf_sq::index<uint8_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | [`const cuvs::neighbors::ivf_sq::index<uint8_t>&`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index) | IVF-SQ index |

**Returns**

`void`

<a id="neighbors-ivf-sq-deserialize"></a>
### neighbors::ivf_sq::deserialize

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_sq::index<uint8_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | [`cuvs::neighbors::ivf_sq::index<uint8_t>*`](/api-reference/cpp-api-neighbors-ivf-sq#neighbors-ivf-sq-index) | IVF-SQ index |

**Returns**

`void`
