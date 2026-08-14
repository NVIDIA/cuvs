---
slug: api-reference/cpp-api-neighbors-ivf-rabitq
---

# IVF RaBitQ

_Source header: `cuvs/neighbors/ivf_rabitq.hpp`_

## IVF-RaBitQ index build parameters

<a id="neighbors-ivf-rabitq-index-params"></a>
### neighbors::ivf_rabitq::index_params

IVF-RaBitQ index build parameters

```cpp
struct index_params : cuvs::neighbors::index_params {
  uint32_t n_lists;
  uint32_t bits_per_dim;
  uint32_t kmeans_n_iters;
  uint32_t max_train_points_per_cluster;
  bool fast_quantize_flag;
  size_t streaming_batch_size;
  bool force_streaming;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_lists` | `uint32_t` | The number of inverted lists (clusters)<br /><br />Hint: Increasing this parameter may alleviate shared memory pressure. |
| `bits_per_dim` | `uint32_t` | The total number of bits per dimension (single bit required for the binary RaBitQ algorithm + additional bits for extended RaBitQ).<br /><br />Supported values: [1, 2, 3, 4, 5, 6, 7, 8, 9].<br /><br />Hint: the smaller the 'bits_per_dim', the smaller the index size and the better the search performance, but the lower the recall. |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (index building). |
| `max_train_points_per_cluster` | `uint32_t` | The number of data vectors (per cluster) to use during iterative kmeans building. |
| `fast_quantize_flag` | `bool` | Flag for using the fast quantize method |
| `streaming_batch_size` | `size_t` | Maximum number of vectors per batch when using streaming construction.<br /><br />This parameter controls the batch size during streaming construction from host memory. Batches contain complete clusters only (no partial clusters across batch boundaries).<br /><br />Note: Streaming construction is automatically used when the dataset doesn't fit comfortably in GPU memory (determined by available workspace and kTolerableRatio). |
| `force_streaming` | `bool` | Force streaming construction regardless of dataset size.<br /><br />When set to true, streaming construction will be used even if the dataset would fit in GPU memory. This is useful for testing or when you want explicit control over the construction method.<br /><br />Note: This parameter only applies when the input dataset is in host memory. If the dataset is already in device memory, streaming construction is not applicable and this parameter has no effect.<br /><br />Default: false (auto-detect based on available memory) |

## IVF-RaBitQ index search parameters

<a id="neighbors-ivf-rabitq-search-mode"></a>
### neighbors::ivf_rabitq::search_mode

A type for specifying the mode for searching the RaBitQ index.

```cpp
enum class search_mode {
  LUT16 = 0,
  LUT32 = 1,
  QUANT4 = 2,
  QUANT8 = 3
};
```

**Values**

| Name | Value |
| --- | --- |
| `LUT16` | `0` |
| `LUT32` | `1` |
| `QUANT4` | `2` |
| `QUANT8` | `3` |

## IVF-RaBitQ index

<a id="neighbors-ivf-rabitq-index"></a>
### neighbors::ivf_rabitq::index

IVF-RaBitQ index.

```cpp
template <typename IdxT>
struct index;
```

<a id="neighbors-ivf-rabitq-index-index"></a>
### neighbors::ivf_rabitq::index::index

Construct an empty index yet to be populated.

```cpp
index(raft::resources const& handle);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_rabitq::index::index`

Construct an empty index yet to be populated.

```cpp
index(raft::resources const& handle,
size_t n_rows,
uint32_t dim,
uint32_t n_lists,
uint32_t bits_per_dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `n_rows` |  | `size_t` |  |
| `dim` |  | `uint32_t` |  |
| `n_lists` |  | `uint32_t` |  |
| `bits_per_dim` |  | `uint32_t` |  |

**Returns**

`void`

<a id="neighbors-ivf-rabitq-index-dim"></a>
### neighbors::ivf_rabitq::index::dim

Dimensionality of the input data.

```cpp
uint32_t dim() const noexcept;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-rabitq-index-size"></a>
### neighbors::ivf_rabitq::index::size

Total length of the index.

```cpp
IdxT size() const noexcept;
```

**Returns**

`IdxT`

<a id="neighbors-ivf-rabitq-index-rabitq-index"></a>
### neighbors::ivf_rabitq::index::rabitq_index

Accessor for underlying RaBitQ index

```cpp
detail::IVFGPU& rabitq_index() noexcept;
```

**Returns**

`detail::IVFGPU&`

## IVF-RaBitQ index build

<a id="neighbors-ivf-rabitq-build"></a>
### neighbors::ivf_rabitq::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_rabitq::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_rabitq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | [`const cuvs::neighbors::ivf_rabitq::index_params&`](/api-reference/cpp-api-neighbors-ivf-rabitq#neighbors-ivf-rabitq-index-params) | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_rabitq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-rabitq#neighbors-ivf-rabitq-index)

**Additional overload:** `neighbors::ivf_rabitq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_rabitq::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_rabitq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | [`const cuvs::neighbors::ivf_rabitq::index_params&`](/api-reference/cpp-api-neighbors-ivf-rabitq#neighbors-ivf-rabitq-index-params) | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_rabitq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-rabitq#neighbors-ivf-rabitq-index)

## IVF-RaBitQ index search

<a id="neighbors-ivf-rabitq-search"></a>
### neighbors::ivf_rabitq::search

Search ANN using the constructed index.

```cpp
void search(raft::resources const& handle,
const cuvs::neighbors::ivf_rabitq::search_params& search_params,
cuvs::neighbors::ivf_rabitq::index<int64_t>& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `search_params` | in | `const cuvs::neighbors::ivf_rabitq::search_params&` | configure the search |
| `index` | in | [`cuvs::neighbors::ivf_rabitq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-rabitq#neighbors-ivf-rabitq-index) | ivf-rabitq constructed index |
| `queries` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | a device matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | a device matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

## IVF-RaBitQ index serialize

<a id="neighbors-ivf-rabitq-serialize"></a>
### neighbors::ivf_rabitq::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_rabitq::index<int64_t>& index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | [`cuvs::neighbors::ivf_rabitq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-rabitq#neighbors-ivf-rabitq-index) | IVF-RaBitQ index |

**Returns**

`void`

<a id="neighbors-ivf-rabitq-deserialize"></a>
### neighbors::ivf_rabitq::deserialize

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_rabitq::index<int64_t>* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | [`cuvs::neighbors::ivf_rabitq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-rabitq#neighbors-ivf-rabitq-index) | IVF-PQ index |

**Returns**

`void`
