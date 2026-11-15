---
slug: api-reference/cpp-api-neighbors-ivf-binary
---

# IVF Binary

_Source header: `cuvs/neighbors/ivf_binary.hpp`_

## IVF-Binary index build parameters

<a id="neighbors-ivf-binary-index-params"></a>
### neighbors::ivf_binary::index_params

IVF-Binary index build parameters

```cpp
struct index_params {
  uint32_t n_lists;
  uint32_t kmeans_n_iters;
  double kmeans_trainset_fraction;
  bool adaptive_centers;
  bool conservative_memory_allocation;
  bool add_data_on_build;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_lists` | `uint32_t` | The number of inverted lists (clusters). |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for k-means centers. |
| `kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative k-means building. |
| `adaptive_centers` | `bool` | By default (adaptive_centers = false), the cluster centers are trained in `build` and never modified in `extend`.<br /><br />With adaptive_centers = true, `extend` updates the centers to the majority bit of the vectors added to each cluster, with ties resolved to 0. The index keeps exact per-bit counts of ones to make the update independent of how the data are split between `extend` calls. |
| `conservative_memory_allocation` | `bool` | Allocate the minimum amount of memory required for the inverted lists, instead of reserving extra space to amortize repeated calls of `extend`. |
| `add_data_on_build` | `bool` | Whether to add the dataset content to the index in `build`. If false, `build` only trains the clustering and the index is populated later with `extend`. |

## IVF-Binary index search parameters

<a id="neighbors-ivf-binary-search-params"></a>
### neighbors::ivf_binary::search_params

IVF-Binary index search parameters

```cpp
struct search_params : cuvs::neighbors::search_params {
  uint32_t n_probes;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |

## IVF-Binary index

<a id="neighbors-ivf-binary-index"></a>
### neighbors::ivf_binary::index

IVF index of bit-packed binary vectors searched with the Hamming distance.

The index stores the codes of a binary-quantized dataset (see `device_binary_dataset_view`), together with the quantizer that produced them. Queries must be supplied as bit-packed bytes. Neighbors are returned with their Hamming distances, i.e. the number of differing bits.

```cpp
template <typename IdxT = int64_t>
struct index;
```

<a id="neighbors-ivf-binary-index-index"></a>
### neighbors::ivf_binary::index::index

Construct an empty index. It needs to be trained with `build` or loaded with `deserialize`.

```cpp
explicit index(raft::resources const& res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |

**Returns**

`explicit`

**Additional overload:** `neighbors::ivf_binary::index::index`

Construct an empty index of `dim`-bit vectors. It needs to be trained and then populated.

```cpp
index(raft::resources const& res, const index_params& params, uint32_t dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`const index_params&`](/api-reference/cpp-api-neighbors-ivf-binary#neighbors-ivf-binary-index-params) |  |
| `dim` |  | `uint32_t` |  |

**Returns**

`void`

<a id="neighbors-ivf-binary-index-dim"></a>
### neighbors::ivf_binary::index::dim

The number of bits in each vector.

```cpp
uint32_t dim() const noexcept;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-binary-index-storage-dim"></a>
### neighbors::ivf_binary::index::storage_dim

The number of bytes used to store each vector.

```cpp
uint32_t storage_dim() const noexcept;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-binary-index-n-lists"></a>
### neighbors::ivf_binary::index::n_lists

The number of clusters/inverted lists.

```cpp
uint32_t n_lists() const noexcept;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-binary-index-size"></a>
### neighbors::ivf_binary::index::size

The total number of vectors in the index.

```cpp
IdxT size() const noexcept;
```

**Returns**

`IdxT`

<a id="neighbors-ivf-binary-index-adaptive-centers"></a>
### neighbors::ivf_binary::index::adaptive_centers

Whether `centers()` change upon extending the index.

```cpp
bool adaptive_centers() const noexcept;
```

**Returns**

`bool`

<a id="neighbors-ivf-binary-index-conservative-memory-allocation"></a>
### neighbors::ivf_binary::index::conservative_memory_allocation

Whether the lists are allocated without extra space (see index_params).

```cpp
bool conservative_memory_allocation() const noexcept;
```

**Returns**

`bool`

<a id="neighbors-ivf-binary-index-list-sizes"></a>
### neighbors::ivf_binary::index::list_sizes

Sizes of the lists (clusters) [n_lists].

```cpp
raft::device_vector_view<const uint32_t, uint32_t> list_sizes() const noexcept;
```

**Returns**

`raft::device_vector_view<const uint32_t, uint32_t>`

<a id="neighbors-ivf-binary-index-centers"></a>
### neighbors::ivf_binary::index::centers

Bit-packed cluster centers [n_lists, storage_dim].

```cpp
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> centers() const noexcept;
```

**Returns**

`raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>`

<a id="neighbors-ivf-binary-index-center-counts"></a>
### neighbors::ivf_binary::index::center_counts

Exact number of ones at each bit of the vectors in each cluster [n_lists, 8 * storage_dim];

```cpp
raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> center_counts() const noexcept;
```

empty unless the centers are adaptive.

**Returns**

`raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>`

<a id="neighbors-ivf-binary-index-quantizer"></a>
### neighbors::ivf_binary::index::quantizer

The build dataset's quantizer, available for external encoding of queries or new vectors.

```cpp
const preprocessing::quantize::binary::quantizer<float>& quantizer() const noexcept;
```

**Returns**

[`const preprocessing::quantize::binary::quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-binary#preprocessing-quantize-binary-quantizer)

## IVF-Binary index build

<a id="neighbors-ivf-binary-build"></a>
### neighbors::ivf_binary::build

Build the index from a binary-quantized dataset.

```cpp
auto build(raft::resources const& handle,
const index_params& index_params,
device_binary_dataset_view<float, int64_t> dataset) -> index<int64_t>;
```

The quantizer of the dataset manifest (the thresholds and the dimension) is copied into the index.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-ivf-binary#neighbors-ivf-binary-index-params) | configure the index building |
| `dataset` | in | `device_binary_dataset_view<float, int64_t>` | the binary-quantized vectors |

**Returns**

[`index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-binary#neighbors-ivf-binary-index)

<a id="neighbors-ivf-binary-extend"></a>
### neighbors::ivf_binary::extend

Add bit-packed vectors to the index.

```cpp
void extend(raft::resources const& handle,
device_binary_dataset_view<float, int64_t> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
index<int64_t>* index);
```

The supplied codes are used as-is. The dictionary of the new vectors is not compared with or copied into the index; choosing the quantizer is the caller's responsibility.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `device_binary_dataset_view<float, int64_t>` | the binary-quantized vectors to add |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | ids of the new vectors; required if the index is not empty, otherwise the ids start from zero |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-binary#neighbors-ivf-binary-index) | the index to extend |

**Returns**

`void`

## IVF-Binary index serialize

<a id="neighbors-ivf-binary-serialize"></a>
### neighbors::ivf_binary::serialize

Save the index to an output stream.

```cpp
void serialize(raft::resources const& handle, std::ostream& os, const index<int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `index` |  | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-binary#neighbors-ivf-binary-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_binary::serialize`

Save the index to a file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const index<int64_t>& index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-binary#neighbors-ivf-binary-index) |  |

**Returns**

`void`

<a id="neighbors-ivf-binary-deserialize"></a>
### neighbors::ivf_binary::deserialize

Load the index from an input stream.

```cpp
void deserialize(raft::resources const& handle, std::istream& is, index<int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `is` |  | `std::istream&` |  |
| `index` |  | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-binary#neighbors-ivf-binary-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_binary::deserialize`

Load the index from a file.

```cpp
void deserialize(raft::resources const& handle, const std::string& filename, index<int64_t>* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `filename` |  | `const std::string&` |  |
| `index` |  | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-binary#neighbors-ivf-binary-index) |  |

**Returns**

`void`
