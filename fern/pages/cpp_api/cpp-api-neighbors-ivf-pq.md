---
slug: api-reference/cpp-api-neighbors-ivf-pq
---

# IVF PQ

_Source header: `cuvs/neighbors/ivf_pq.hpp`_

## IVF-PQ index build parameters

<a id="neighbors-ivf-pq-codebook-gen"></a>
### neighbors::ivf_pq::codebook_gen

A type for specifying how PQ codebooks are created.

```cpp
enum class codebook_gen;
```

<a id="neighbors-ivf-pq-list-layout"></a>
### neighbors::ivf_pq::list_layout

A type for specifying the memory layout of PQ codes in IVF lists.

```cpp
enum class list_layout;
```

<a id="neighbors-ivf-pq-index-params-from-dataset"></a>
### neighbors::ivf_pq::index_params::from_dataset

Creates index_params based on shape of the input dataset.

```cpp
static index_params from_dataset(
raft::matrix_extent<int64_t> dataset,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dataset` |  | `raft::matrix_extent<int64_t>` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`static index_params`

## IVF-PQ index search parameters

<a id="neighbors-ivf-pq-search-params"></a>
### neighbors::ivf_pq::search_params

IVF-PQ index search parameters

```cpp
struct search_params : cuvs::neighbors::search_params {
  uint32_t n_probes;
  cudaDataType_t lut_dtype;
  cudaDataType_t internal_distance_dtype;
  double preferred_shmem_carveout;
  cudaDataType_t coarse_search_dtype;
  uint32_t max_internal_batch_size;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |
| `lut_dtype` | `cudaDataType_t` | Data type of look up table to be created dynamically at search time.<br /><br />Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]<br /><br />The use of low-precision types reduces the amount of shared memory required at search time, so fast shared memory kernels can be used even for datasets with large dimansionality. Note that the recall is slightly degraded when low-precision type is selected. |
| `internal_distance_dtype` | `cudaDataType_t` | Storage data type for distance/similarity computed at search time.<br /><br />Possible values: [CUDA_R_16F, CUDA_R_32F]<br /><br />If the performance limiter at search time is device memory access, selecting FP16 will improve performance slightly. |
| `preferred_shmem_carveout` | `double` | Preferred fraction of SM's unified memory / L1 cache to be used as shared memory.<br /><br />Possible values: [0.0 - 1.0] as a fraction of the `sharedMemPerMultiprocessor`.<br /><br />One wants to increase the carveout to make sure a good GPU occupancy for the main search kernel, but not to keep it too high to leave some memory to be used as L1 cache. Note, this value is interpreted only as a hint. Moreover, a GPU usually allows only a fixed set of cache configurations, so the provided value is rounded up to the nearest configuration. Refer to the NVIDIA tuning guide for the target GPU architecture.<br /><br />Note, this is a low-level tuning parameter that can have drastic negative effects on the search performance if tweaked incorrectly. |
| `coarse_search_dtype` | `cudaDataType_t` | [Experimental] The data type to use as the GEMM element type when searching the clusters to probe.<br /><br />Possible values: [CUDA_R_8I, CUDA_R_16F, CUDA_R_32F].<br /><br />- Legacy default: CUDA_R_32F (float)<br />- Recommended for performance: CUDA_R_16F (half)<br />- Experimental/low-precision: CUDA_R_8I (int8_t) (WARNING: int8_t variant degrades recall unless data is normalized and low-dimensional) |
| `max_internal_batch_size` | `uint32_t` | Set the internal batch size to improve GPU utilization at the cost of larger memory footprint. |

## Types

<a id="list-extents"></a>
### list_extents

PQ-encoded data stored in the interleaved format:

```cpp
using list_extents = raft::
extents<SizeT, raft::dynamic_extent, raft::dynamic_extent, kIndexGroupSize, kIndexGroupVecLen>;
```

<a id="neighbors-ivf-pq-list-spec-flat"></a>
### neighbors::ivf_pq::list_spec_flat

Flat (non-interleaved) storage specification for PQ-encoded data.

This stores each vector's PQ codes contiguously: [n_rows, bytes_per_vector] where bytes_per_vector = ceildiv(pq_dim * pq_bits, 8)

```cpp
template <typename SizeT, typename IdxT>
struct list_spec_flat {
  SizeT align_max;
  SizeT align_min;
  uint32_t pq_bits;
  uint32_t pq_dim;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `align_max` | `SizeT` |  |
| `align_min` | `SizeT` |  |
| `pq_bits` | `uint32_t` |  |
| `pq_dim` | `uint32_t` |  |

<a id="neighbors-graph-build-params-ivf-pq-params"></a>
### neighbors::graph_build_params::ivf_pq_params

Specialized parameters utilizing IVF-PQ to build knn graph

```cpp
struct ivf_pq_params {
  cuvs::neighbors::ivf_pq::index_params build_params;
  cuvs::neighbors::ivf_pq:: search_params;
  float refinement_rate;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `build_params` | `cuvs::neighbors::ivf_pq::index_params` |  |
| `search_params` | `cuvs::neighbors::ivf_pq::` |  |
| `refinement_rate` | `float` |  |

## IVF-PQ index

<a id="neighbors-ivf-pq-index"></a>
### neighbors::ivf_pq::index

IVF-PQ index.

In the IVF-PQ index, a database vector y is approximated with two level quantization:

y = Q_1(y) + Q_2(y - Q_1(y))

The first level quantizer (Q_1), maps the vector y to the nearest cluster center. The number of clusters is n_lists.

The second quantizer encodes the residual, and it is defined as a product quantizer [1].

A product quantizer encodes a `dim` dimensional vector with a `pq_dim` dimensional vector. First we split the input vector into `pq_dim` subvectors (denoted by u), where each u vector contains `pq_len` distinct components of y

y_1, y_2, ... y_\{pq_len\}, y_\{pq_len+1\}, ... y_\{2*pq_len\}, ... y_\{dim-pq_len+1\} ... y_\{dim\} u_1                         u_2                          u_\{pq_dim\}

Then each subvector encoded with a separate quantizer q_i, end the results are concatenated

Q_2(y) = q_1(u_1),q_2(u_2),...,q_\{pq_dim\}(u_pq_dim\})

Each quantizer q_i outputs a code with pq_bit bits. The second level quantizers are also defined by k-means clustering in the corresponding sub-space: the reproduction values are the centroids, and the set of reproduction values is the codebook.

When the data dimensionality `dim` is not multiple of `pq_dim`, the feature space is transformed using a random orthogonal matrix to have `rot_dim = pq_dim * pq_len` dimensions (`rot_dim &gt;= dim`).

The second-level quantizers are trained either for each subspace or for each cluster: (a) codebook_gen::PER_SUBSPACE: creates `pq_dim` second-level quantizers - one for each slice of the data along features; (b) codebook_gen::PER_CLUSTER: creates `n_lists` second-level quantizers - one for each first-level cluster. In either case, the centroids are again found using k-means clustering interpreting the data as having pq_len dimensions.

[1] Product quantization for nearest neighbor search Herve Jegou, Matthijs Douze, Cordelia Schmid

```cpp
template <typename IdxT>
class index;
```

<a id="neighbors-ivf-pq-index-index"></a>
### neighbors::ivf_pq::index::index

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

**Additional overload:** `neighbors::ivf_pq::index::index`

Construct an index with specified parameters.

```cpp
index(raft::resources const& handle,
cuvs::distance::DistanceType metric,
codebook_gen codebook_kind,
uint32_t n_lists,
uint32_t dim,
uint32_t pq_bits                    = 8,
uint32_t pq_dim                     = 0,
bool conservative_memory_allocation = false);
```

This constructor creates an owning index with the given parameters.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` | RAFT resources handle |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Distance metric for clustering |
| `codebook_kind` |  | [`codebook_gen`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-codebook-gen) | How PQ codebooks are created |
| `n_lists` |  | `uint32_t` | Number of inverted lists (clusters) |
| `dim` |  | `uint32_t` | Dimensionality of the input data |
| `pq_bits` |  | `uint32_t` | Bit length of vector elements after PQ compression<br />Default: `8`. |
| `pq_dim` |  | `uint32_t` | Dimensionality after PQ compression (0 = auto-select)<br />Default: `0`. |
| `conservative_memory_allocation` |  | `bool` | Memory allocation strategy<br />Default: `false`. |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::index::index`

Construct an index from index parameters.

```cpp
index(raft::resources const& handle, const index_params& params, uint32_t dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` | RAFT resources handle |
| `params` |  | `const index_params&` | Index parameters |
| `dim` |  | `uint32_t` | Dimensionality of the input data |

**Returns**

`void`

<a id="neighbors-ivf-pq-index-size"></a>
### neighbors::ivf_pq::index::size

Total length of the index.

```cpp
IdxT size() const noexcept override;
```

**Returns**

`IdxT`

<a id="neighbors-ivf-pq-index-dim"></a>
### neighbors::ivf_pq::index::dim

Dimensionality of the input data.

```cpp
uint32_t dim() const noexcept override;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-pq-index-dim-ext"></a>
### neighbors::ivf_pq::index::dim_ext

Dimensionality of the cluster centers:

```cpp
uint32_t dim_ext() const noexcept;
```

input data dim extended with vector norms and padded to 8 elems.

**Returns**

`uint32_t`

<a id="neighbors-ivf-pq-index-rot-dim"></a>
### neighbors::ivf_pq::index::rot_dim

Dimensionality of the data after transforming it for PQ processing

```cpp
uint32_t rot_dim() const noexcept;
```

(rotated and augmented to be muplitple of `pq_dim`).

**Returns**

`uint32_t`

<a id="neighbors-ivf-pq-index-pq-bits"></a>
### neighbors::ivf_pq::index::pq_bits

The bit length of an encoded vector element after compression by PQ.

```cpp
uint32_t pq_bits() const noexcept override;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-pq-index-pq-dim"></a>
### neighbors::ivf_pq::index::pq_dim

The dimensionality of an encoded vector after compression by PQ.

```cpp
uint32_t pq_dim() const noexcept override;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-pq-index-pq-len"></a>
### neighbors::ivf_pq::index::pq_len

Dimensionality of a subspace, i.e. the number of vector components mapped to a subspace

```cpp
uint32_t pq_len() const noexcept;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-pq-index-pq-book-size"></a>
### neighbors::ivf_pq::index::pq_book_size

The number of vectors in a PQ codebook (`1 &lt;&lt; pq_bits`).

```cpp
uint32_t pq_book_size() const noexcept;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-pq-index-metric"></a>
### neighbors::ivf_pq::index::metric

Distance metric used for clustering.

```cpp
cuvs::distance::DistanceType metric() const noexcept override;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="neighbors-ivf-pq-index-codebook-kind"></a>
### neighbors::ivf_pq::index::codebook_kind

How PQ codebooks are created.

```cpp
codebook_gen codebook_kind() const noexcept override;
```

**Returns**

[`codebook_gen`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-codebook-gen)

<a id="neighbors-ivf-pq-index-codes-layout"></a>
### neighbors::ivf_pq::index::codes_layout

Memory layout of PQ codes in IVF lists.

```cpp
list_layout codes_layout() const noexcept override;
```

**Returns**

[`list_layout`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-list-layout)

<a id="neighbors-ivf-pq-index-n-lists"></a>
### neighbors::ivf_pq::index::n_lists

Number of clusters/inverted lists (first level quantization).

```cpp
uint32_t n_lists() const noexcept;
```

**Returns**

`uint32_t`

<a id="neighbors-ivf-pq-index-conservative-memory-allocation"></a>
### neighbors::ivf_pq::index::conservative_memory_allocation

Whether to use conservative memory allocation when extending the list (cluster) data

```cpp
bool conservative_memory_allocation() const noexcept override;
```

(see index_params.conservative_memory_allocation).

**Returns**

`bool`

<a id="neighbors-ivf-pq-index-pq-centers"></a>
### neighbors::ivf_pq::index::pq_centers

PQ cluster centers

```cpp
raft::device_mdspan<const float, pq_centers_extents, raft::row_major> pq_centers()
const noexcept override;
```

- codebook_gen::PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
- codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]

**Returns**

`raft::device_mdspan<const float, pq_centers_extents, raft::row_major>`

<a id="neighbors-ivf-pq-index-lists"></a>
### neighbors::ivf_pq::index::lists

Lists' data and indices (polymorphic, works for both FLAT and INTERLEAVED layouts).

```cpp
std::vector<std::shared_ptr<list_data_base<IdxT>>>& lists() noexcept override;
```

**Returns**

`std::vector<std::shared_ptr<list_data_base<IdxT>>>&`

<a id="neighbors-ivf-pq-index-data-ptrs"></a>
### neighbors::ivf_pq::index::data_ptrs

Pointers to the inverted lists (clusters) data  [n_lists].

```cpp
raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs() noexcept override;
```

**Returns**

`raft::device_vector_view<uint8_t*, uint32_t, raft::row_major>`

<a id="neighbors-ivf-pq-index-inds-ptrs"></a>
### neighbors::ivf_pq::index::inds_ptrs

Pointers to the inverted lists (clusters) indices  [n_lists].

```cpp
raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs() noexcept override;
```

**Returns**

`raft::device_vector_view<IdxT*, uint32_t, raft::row_major>`

<a id="neighbors-ivf-pq-index-rotation-matrix"></a>
### neighbors::ivf_pq::index::rotation_matrix

The transform matrix (original space -&gt; rotated padded space) [rot_dim, dim]

```cpp
raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix()
const noexcept override;
```

**Returns**

`raft::device_matrix_view<const float, uint32_t, raft::row_major>`

<a id="neighbors-ivf-pq-index-accum-sorted-sizes"></a>
### neighbors::ivf_pq::index::accum_sorted_sizes

Accumulated list sizes, sorted in descending order [n_lists + 1].

```cpp
raft::host_vector_view<IdxT, uint32_t, raft::row_major> accum_sorted_sizes() noexcept override;
```

The last value contains the total length of the index. The value at index zero is always zero.

That is, the content of this span is as if the `list_sizes` was sorted and then accumulated.

This span is used during search to estimate the maximum size of the workspace.

**Returns**

`raft::host_vector_view<IdxT, uint32_t, raft::row_major>`

<a id="neighbors-ivf-pq-index-list-sizes"></a>
### neighbors::ivf_pq::index::list_sizes

Sizes of the lists [n_lists].

```cpp
raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes() noexcept override;
```

**Returns**

`raft::device_vector_view<uint32_t, uint32_t, raft::row_major>`

<a id="neighbors-ivf-pq-index-centers"></a>
### neighbors::ivf_pq::index::centers

Cluster centers corresponding to the lists in the original space [n_lists, dim_ext]

```cpp
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers()
const noexcept override;
```

**Returns**

`raft::device_matrix_view<const float, uint32_t, raft::row_major>`

<a id="neighbors-ivf-pq-index-centers-rot"></a>
### neighbors::ivf_pq::index::centers_rot

Cluster centers corresponding to the lists in the rotated space [n_lists, rot_dim]

```cpp
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot()
const noexcept override;
```

**Returns**

`raft::device_matrix_view<const float, uint32_t, raft::row_major>`

<a id="neighbors-ivf-pq-index-get-list-size-in-bytes"></a>
### neighbors::ivf_pq::index::get_list_size_in_bytes

fetch size of a particular IVF list in bytes using the list extents.

```cpp
uint32_t get_list_size_in_bytes(uint32_t label) const override;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `label` | in | `uint32_t` | list ID |

**Returns**

`uint32_t`

**Additional overload:** `neighbors::ivf_pq::index::index`

Construct index from implementation pointer.

```cpp
explicit index(std::unique_ptr<index_iface<IdxT>> impl);
```

This constructor is used internally by build/extend/deserialize functions.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `impl` |  | `std::unique_ptr<index_iface<IdxT>>` | Implementation pointer (owning or view) |

**Returns**

`explicit`

## IVF-PQ index build

<a id="neighbors-ivf-pq-build"></a>
### neighbors::ivf_pq::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | reference to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | reference to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | reference to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | reference to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | reference to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | reference to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | reference to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | reference to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build a view-type IVF-PQ index from device memory centroids and codebook.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
const uint32_t dim,
raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,
raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

This function creates a non-owning index that stores a reference to the provided device data. All parameters must be provided with correct extents. The caller is responsible for ensuring the lifetime of the input data exceeds the lifetime of the returned index.

The index_params must be consistent with the provided matrices. Specifically:

- index_params.codebook_kind determines the expected shape of pq_centers
- index_params.metric will be stored in the index
- index_params.conservative_memory_allocation will be stored in the index The function will verify consistency between index_params, dim, and the matrix extents.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources handle |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index (metric, codebook_kind, etc.). Must be consistent with the provided matrices. |
| `dim` | in | `const uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>` | PQ codebook on device memory with required extents:<br />- codebook_gen::PER_SUBSPACE: [pq_dim, pq_len, pq_book_size]<br />- codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size] |
| `centers` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Cluster centers in the original space [n_lists, dim_ext] where dim_ext = round_up(dim + 1, 8) |
| `centers_rot` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Rotated cluster centers [n_lists, rot_dim] where rot_dim = pq_len * pq_dim |
| `rotation_matrix` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Transform matrix (original space -&gt; rotated padded space) [rot_dim, dim] |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build an IVF-PQ index from device memory centroids and codebook.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
const uint32_t dim,
raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,
raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

This function creates a non-owning index that references the provided device data directly. All parameters must be provided with correct extents. The caller is responsible for ensuring the lifetime of the input data exceeds the lifetime of the returned index.

The index_params must be consistent with the provided matrices. Specifically:

- index_params.codebook_kind determines the expected shape of pq_centers
- index_params.metric will be stored in the index
- index_params.conservative_memory_allocation will be stored in the index The function will verify consistency between index_params, dim, and the matrix extents.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources handle |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index (metric, codebook_kind, etc.). Must be consistent with the provided matrices. |
| `dim` | in | `const uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>` | PQ codebook on device memory with required extents:<br />- codebook_gen::PER_SUBSPACE: [pq_dim, pq_len, pq_book_size]<br />- codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size] |
| `centers` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Cluster centers in the original space [n_lists, dim_ext] where dim_ext = round_up(dim + 1, 8) |
| `centers_rot` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Rotated cluster centers [n_lists, rot_dim] where rot_dim = pq_len * pq_dim |
| `rotation_matrix` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Transform matrix (original space -&gt; rotated padded space) [rot_dim, dim] |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | pointer to ivf_pq::index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::build`

Build an IVF-PQ index from host memory centroids and codebook (in-place).

```cpp
auto build(
raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
const uint32_t dim,
raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,
std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources handle |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dim` | in | `const uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>` | PQ codebook |
| `centers` | in | `raft::host_matrix_view<const float, uint32_t, raft::row_major>` | Cluster centers |
| `centers_rot` | in | `std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>>` | Optional rotated cluster centers |
| `rotation_matrix` | in | `std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>>` | Optional rotation matrix |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::build`

Build an IVF-PQ index from host memory centroids and codebook (in-place).

```cpp
void build(
raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
const uint32_t dim,
raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,
std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources handle |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dim` | in | `const uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>` | PQ codebook on host memory |
| `centers` | in | `raft::host_matrix_view<const float, uint32_t, raft::row_major>` | Cluster centers on host memory |
| `centers_rot` | in | `std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>>` | Optional rotated cluster centers on host |
| `rotation_matrix` | in | `std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>>` | Optional rotation matrix on host |
| `idx` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | pointer to IVF-PQ index to be built |

**Returns**

`void`

## IVF-PQ index extend

<a id="neighbors-ivf-pq-extend"></a>
### neighbors::ivf_pq::extend

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

[`cuvs::neighbors::ivf_pq::index<int64_t>`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index)

**Additional overload:** `neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |

**Returns**

`void`

## IVF-PQ index transform

<a id="neighbors-ivf-pq-transform"></a>
### neighbors::ivf_pq::transform

Transform a dataset by applying pq-encoding to each vector

```cpp
void transform(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index<int64_t>& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
raft::device_vector_view<uint32_t, int64_t> output_labels,
raft::device_matrix_view<uint8_t, int64_t> output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index` | in | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | ivf-pq constructed index |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, index.dim()] |
| `output_labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | a device vector view [n_rows] that will get populaterd with the cluster ids (labels) for each vector in the input dataset |
| `output_dataset` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a device matrix view [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)]] that will get populated with the pq-encoded dataset |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::transform`

```cpp
void transform(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index<int64_t>& index,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
raft::device_vector_view<uint32_t, int64_t> output_labels,
raft::device_matrix_view<uint8_t, int64_t> output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `index` |  | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `dataset` |  | `raft::device_matrix_view<const half, int64_t, raft::row_major>` |  |
| `output_labels` |  | `raft::device_vector_view<uint32_t, int64_t>` |  |
| `output_dataset` |  | `raft::device_matrix_view<uint8_t, int64_t>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::transform`

```cpp
void transform(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index<int64_t>& index,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
raft::device_vector_view<uint32_t, int64_t> output_labels,
raft::device_matrix_view<uint8_t, int64_t> output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `index` |  | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `dataset` |  | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` |  |
| `output_labels` |  | `raft::device_vector_view<uint32_t, int64_t>` |  |
| `output_dataset` |  | `raft::device_matrix_view<uint8_t, int64_t>` |  |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::transform`

```cpp
void transform(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index<int64_t>& index,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
raft::device_vector_view<uint32_t, int64_t> output_labels,
raft::device_matrix_view<uint8_t, int64_t> output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `index` |  | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `dataset` |  | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` |  |
| `output_labels` |  | `raft::device_vector_view<uint32_t, int64_t>` |  |
| `output_dataset` |  | `raft::device_matrix_view<uint8_t, int64_t>` |  |

**Returns**

`void`

## IVF-PQ index serialize

<a id="neighbors-ivf-pq-serialize"></a>
### neighbors::ivf_pq::serialize

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::ivf_pq::index<int64_t>& index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | IVF-PQ index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::ivf_pq::index<int64_t>& index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | [`const cuvs::neighbors::ivf_pq::index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | IVF-PQ index |

**Returns**

`void`

<a id="neighbors-ivf-pq-deserialize"></a>
### neighbors::ivf_pq::deserialize

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& str,
cuvs::neighbors::ivf_pq::index<int64_t>* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `str` | in | `std::istream&` | the name of the file that stores the index |
| `index` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | IVF-PQ index |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_pq::index<int64_t>* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | [`cuvs::neighbors::ivf_pq::index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | IVF-PQ index |

**Returns**

`void`

## IVF-PQ helper methods

<a id="neighbors-ivf-pq-helpers-codepacker-unpack"></a>
### neighbors::ivf_pq::helpers::codepacker::unpack

Unpack `n_take` consecutive records of a single list (cluster) in the compressed index starting at given `offset`.

```cpp
void unpack(raft::resources const& res,
raft::device_mdspan<const uint8_t,
list_spec_interleaved<uint32_t, uint32_t>::list_extents,
raft::row_major> list_data,
uint32_t pq_bits,
uint32_t offset,
raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes);
```

Bit compression is removed, which means output will have pq_dim dimensional vectors (one code per byte, instead of ceildiv(pq_dim * pq_bits, 8) bytes of pq codes).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `list_data` | in | [`raft::device_mdspan<const uint8_t, list_spec_interleaved<uint32_t, uint32_t>::list_extents, raft::row_major>`](/api-reference/cpp-api-neighbors-ivf-pq#list-extents) | block to read from |
| `pq_bits` | in | `uint32_t` | bit length of encoded vector elements |
| `offset` | in | `uint32_t` | How many records in the list to skip. |
| `codes` | out | `raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>` | the destination buffer [n_take, index.pq_dim()]. The length `n_take` defines how many records to unpack, it must be smaller than the list size. |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-unpack-contiguous"></a>
### neighbors::ivf_pq::helpers::codepacker::unpack_contiguous

Unpack `n_rows` consecutive records of a single list (cluster) in the compressed index starting at given `offset`. The output codes of a single vector are contiguous, not expanded to one code per byte, which means the output has ceildiv(pq_dim * pq_bits, 8) bytes per PQ encoded vector.

```cpp
void unpack_contiguous(raft::resources const& res,
raft::device_mdspan<const uint8_t,
list_spec_interleaved<uint32_t, uint32_t>::list_extents,
raft::row_major> list_data,
uint32_t pq_bits,
uint32_t offset,
uint32_t n_rows,
uint32_t pq_dim,
uint8_t* codes);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `list_data` | in | [`raft::device_mdspan<const uint8_t, list_spec_interleaved<uint32_t, uint32_t>::list_extents, raft::row_major>`](/api-reference/cpp-api-neighbors-ivf-pq#list-extents) | block to read from |
| `pq_bits` | in | `uint32_t` | bit length of encoded vector elements |
| `offset` | in | `uint32_t` | How many records in the list to skip. |
| `n_rows` | in | `uint32_t` | How many records to unpack |
| `pq_dim` | in | `uint32_t` | The dimensionality of the PQ compressed records |
| `codes` | out | `uint8_t*` | the destination buffer [n_rows, ceildiv(pq_dim * pq_bits, 8)]. The length `n_rows` defines how many records to unpack, it must be smaller than the list size. |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-pack"></a>
### neighbors::ivf_pq::helpers::codepacker::pack

Write flat PQ codes into an existing list by the given offset.

```cpp
void pack(raft::resources const& res,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
uint32_t pq_bits,
uint32_t offset,
raft::device_mdspan<uint8_t,
list_spec_interleaved<uint32_t, uint32_t>::list_extents,
raft::row_major> list_data);
```

NB: no memory allocation happens here; the list must fit the data (offset + n_vec).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat PQ codes, one code per byte [n_vec, pq_dim] |
| `pq_bits` | in | `uint32_t` | bit length of encoded vector elements |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |
| `list_data` | in | [`raft::device_mdspan<uint8_t, list_spec_interleaved<uint32_t, uint32_t>::list_extents, raft::row_major>`](/api-reference/cpp-api-neighbors-ivf-pq#list-extents) | block to write into |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-pack-contiguous"></a>
### neighbors::ivf_pq::helpers::codepacker::pack_contiguous

Write flat PQ codes into an existing list by the given offset. The input codes of a single vector

```cpp
void pack_contiguous(raft::resources const& res,
const uint8_t* codes,
uint32_t n_rows,
uint32_t pq_dim,
uint32_t pq_bits,
uint32_t offset,
raft::device_mdspan<uint8_t,
list_spec_interleaved<uint32_t, uint32_t>::list_extents,
raft::row_major> list_data);
```

are contiguous (not expanded to one code per byte).

NB: no memory allocation happens here; the list must fit the data (offset + n_rows records).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `codes` | in | `const uint8_t*` | flat PQ codes, [n_vec, ceildiv(pq_dim * pq_bits, 8)] |
| `n_rows` | in | `uint32_t` | number of records |
| `pq_dim` | in | `uint32_t` |  |
| `pq_bits` | in | `uint32_t` | bit length of encoded vector elements |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |
| `list_data` | in | [`raft::device_mdspan<uint8_t, list_spec_interleaved<uint32_t, uint32_t>::list_extents, raft::row_major>`](/api-reference/cpp-api-neighbors-ivf-pq#list-extents) | block to write into |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-pack-list-data"></a>
### neighbors::ivf_pq::helpers::codepacker::pack_list_data

Write flat PQ codes into an existing list by the given offset.

```cpp
void pack_list_data(raft::resources const& res,
index<int64_t>* index,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
uint32_t label,
uint32_t offset);
```

The list is identified by its label.

NB: no memory allocation happens here; the list must fit the data (offset + n_vec).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | IVF-PQ index. |
| `codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat PQ codes, one code per byte [n_rows, pq_dim] |
| `label` | in | `uint32_t` | The id of the list (cluster) into which we write. |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-pack-contiguous-list-data"></a>
### neighbors::ivf_pq::helpers::codepacker::pack_contiguous_list_data

Write flat PQ codes into an existing list by the given offset. Use this when the input

```cpp
void pack_contiguous_list_data(raft::resources const& res,
index<int64_t>* index,
uint8_t* codes,
uint32_t n_rows,
uint32_t label,
uint32_t offset);
```

vectors are PQ encoded and not expanded to one code per byte.

The list is identified by its label.

NB: no memory allocation happens here; the list into which the vectors are packed must fit offset + n_rows rows.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | pointer to IVF-PQ index |
| `codes` | in | `uint8_t*` | flat contiguous PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)] |
| `n_rows` | in | `uint32_t` | how many records to pack |
| `label` | in | `uint32_t` | The id of the list (cluster) into which we write. |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-unpack-list-data"></a>
### neighbors::ivf_pq::helpers::codepacker::unpack_list_data

Unpack `n_take` consecutive records of a single list (cluster) in the compressed index starting at given `offset`, one code per byte (independently of pq_bits).

```cpp
void unpack_list_data(raft::resources const& res,
const index<int64_t>& index,
raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
uint32_t label,
uint32_t offset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | in | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `out_codes` | out | `raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>` | the destination buffer [n_take, index.pq_dim()]. The length `n_take` defines how many records to unpack, it must be smaller than the list size. |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `offset` | in | `uint32_t` | How many records in the list to skip. |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::helpers::codepacker::unpack_list_data`

Unpack a series of records of a single list (cluster) in the compressed index by their in-list offsets, one code per byte (independently of pq_bits).

```cpp
void unpack_list_data(raft::resources const& res,
const index<int64_t>& index,
raft::device_vector_view<const uint32_t> in_cluster_indices,
raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
uint32_t label);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | in | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | IVF-PQ index (passed by reference) |
| `in_cluster_indices` | in | `raft::device_vector_view<const uint32_t>` | The offsets of the selected indices within the cluster. |
| `out_codes` | out | `raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>` | the destination buffer [n_take, index.pq_dim()]. The length `n_take` defines how many records to unpack, it must be smaller than the list size. |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-unpack-contiguous-list-data"></a>
### neighbors::ivf_pq::helpers::codepacker::unpack_contiguous_list_data

Unpack `n_rows` consecutive PQ encoded vectors of a single list (cluster) in the compressed index starting at given `offset`, not expanded to one code per byte. Each code in the output buffer occupies ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes.

```cpp
void unpack_contiguous_list_data(raft::resources const& res,
const index<int64_t>& index,
uint8_t* out_codes,
uint32_t n_rows,
uint32_t label,
uint32_t offset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | in | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | IVF-PQ index (passed by reference) |
| `out_codes` | out | `uint8_t*` | the destination buffer [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)]. The length `n_rows` defines how many records to unpack, offset + n_rows must be smaller than or equal to the list size. |
| `n_rows` | in | `uint32_t` | how many codes to unpack |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `offset` | in | `uint32_t` | How many records in the list to skip. |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-reconstruct-list-data"></a>
### neighbors::ivf_pq::helpers::codepacker::reconstruct_list_data

Decode `n_take` consecutive records of a single list (cluster) in the compressed index starting at given `offset`.

```cpp
void reconstruct_list_data(raft::resources const& res,
const index<int64_t>& index,
raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
uint32_t label,
uint32_t offset);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | in | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `out_vectors` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | the destination buffer [n_take, index.dim()]. The length `n_take` defines how many records to reconstruct, it must be smaller than the list size. |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `offset` | in | `uint32_t` | How many records in the list to skip. |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::helpers::codepacker::reconstruct_list_data`

Decode a series of records of a single list (cluster) in the compressed index by their in-list offsets.

```cpp
void reconstruct_list_data(raft::resources const& res,
const index<int64_t>& index,
raft::device_vector_view<const uint32_t> in_cluster_indices,
raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
uint32_t label);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | in | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `in_cluster_indices` | in | `raft::device_vector_view<const uint32_t>` | The offsets of the selected indices within the cluster. |
| `out_vectors` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | the destination buffer [n_take, index.dim()]. The length `n_take` defines how many records to reconstruct, it must be smaller than the list size. |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-extend-list-with-codes"></a>
### neighbors::ivf_pq::helpers::codepacker::extend_list_with_codes

Extend one list of the index in-place, by the list label, skipping the classification and encoding steps.

```cpp
void extend_list_with_codes(
raft::resources const& res,
index<int64_t>* index,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
uint32_t label);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `new_codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat PQ codes, one code per byte [n_rows, index.pq_dim()] |
| `new_indices` | in | `raft::device_vector_view<const int64_t, uint32_t, raft::row_major>` | source indices [n_rows] |
| `label` | in | `uint32_t` | the id of the target list (cluster). |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-extend-list-with-contiguous-codes"></a>
### neighbors::ivf_pq::helpers::codepacker::extend_list_with_contiguous_codes

Extend one list of the index in-place, by the list label, skipping the classification and encoding steps. Uses contiguous/packed codes format.

```cpp
void extend_list_with_contiguous_codes(
raft::resources const& res,
index<int64_t>* index,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
uint32_t label);
```

This is similar to extend_list_with_codes but takes codes in contiguous packed format [n_rows, ceildiv(pq_dim * pq_bits, 8)] instead of unpacked format [n_rows, pq_dim]. This works correctly with any pq_bits value.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `new_codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat contiguous PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)] |
| `new_indices` | in | `raft::device_vector_view<const int64_t, uint32_t, raft::row_major>` | source indices [n_rows] |
| `label` | in | `uint32_t` | the id of the target list (cluster). |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-codepacker-extend-list"></a>
### neighbors::ivf_pq::helpers::codepacker::extend_list

Extend one list of the index in-place, by the list label, skipping the classification step.

```cpp
void extend_list(raft::resources const& res,
index<int64_t>* index,
raft::device_matrix_view<const float, uint32_t, raft::row_major> new_vectors,
raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
uint32_t label);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | data to encode [n_rows, index.dim()] |
| `new_indices` | in | `raft::device_vector_view<const int64_t, uint32_t, raft::row_major>` | source indices [n_rows] |
| `label` | in | `uint32_t` | the id of the target list (cluster). |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-erase-list"></a>
### neighbors::ivf_pq::helpers::erase_list

Remove all data from a single list (cluster) in the index.

```cpp
void erase_list(raft::resources const& res, index<int64_t>* index, uint32_t label);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `label` | in | `uint32_t` | the id of the target list (cluster). |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-reset-index"></a>
### neighbors::ivf_pq::helpers::reset_index

Public helper API to reset the data and indices ptrs, and the list sizes. Useful for externally modifying the index without going through the build stage. The data and indices of the IVF lists will be lost.

```cpp
void reset_index(const raft::resources& res, index<int64_t>* index);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | pointer to IVF-PQ index |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-pad-centers-with-norms"></a>
### neighbors::ivf_pq::helpers::pad_centers_with_norms

Pad cluster centers with their L2 norms for efficient GEMM operations.

```cpp
void pad_centers_with_norms(
raft::resources const& res,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
raft::device_matrix_view<float, uint32_t, raft::row_major> padded_centers);
```

This function takes cluster centers and pads them with their L2 norms to create extended centers suitable for coarse search operations. The output has dimensions [n_centers, dim_ext] where dim_ext = round_up(dim + 1, 8).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `centers` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | cluster centers [n_centers, dim] |
| `padded_centers` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | padded centers with norms [n_centers, dim_ext] |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::helpers::pad_centers_with_norms`

Pad cluster centers with their L2 norms for efficient GEMM operations.

```cpp
void pad_centers_with_norms(
raft::resources const& res,
raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
raft::device_matrix_view<float, uint32_t, raft::row_major> padded_centers);
```

This function takes cluster centers and pads them with their L2 norms to create extended centers suitable for coarse search operations. The output has dimensions [n_centers, dim_ext] where dim_ext = round_up(dim + 1, 8).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `centers` | in | `raft::host_matrix_view<const float, uint32_t, raft::row_major>` | cluster centers [n_centers, dim] |
| `padded_centers` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | padded centers with norms [n_centers, dim_ext] |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-rotate-padded-centers"></a>
### neighbors::ivf_pq::helpers::rotate_padded_centers

Rotate padded centers with the rotation matrix.

```cpp
void rotate_padded_centers(
raft::resources const& res,
raft::device_matrix_view<const float, uint32_t, raft::row_major> padded_centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix,
raft::device_matrix_view<float, uint32_t, raft::row_major> rotated_centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `padded_centers` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | padded centers [n_centers, dim_ext] |
| `rotation_matrix` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | rotation matrix [rot_dim, dim] |
| `rotated_centers` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | rotated centers [n_centers, rot_dim] |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-extract-centers"></a>
### neighbors::ivf_pq::helpers::extract_centers

Public helper API for fetching a trained index's IVF centroids

```cpp
void extract_centers(raft::resources const& res,
const index<int64_t>& index,
raft::device_matrix_view<float, int64_t, raft::row_major> cluster_centers);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | in | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | IVF-PQ index (passed by reference) |
| `cluster_centers` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | IVF cluster centers [index.n_lists(), index.dim] |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::helpers::extract_centers`

```cpp
void extract_centers(raft::resources const& res,
const index<int64_t>& index,
raft::host_matrix_view<float, uint32_t, raft::row_major> cluster_centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `index` |  | [`const index<int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) |  |
| `cluster_centers` |  | `raft::host_matrix_view<float, uint32_t, raft::row_major>` |  |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-recompute-internal-state"></a>
### neighbors::ivf_pq::helpers::recompute_internal_state

Helper exposing the re-computation of list sizes and related arrays if IVF lists have been modified externally.

```cpp
void recompute_internal_state(const raft::resources& res, index<int64_t>* index);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | [`index<int64_t>*`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-index) | pointer to IVF-PQ index |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-make-rotation-matrix"></a>
### neighbors::ivf_pq::helpers::make_rotation_matrix

Generate a rotation matrix into user-provided buffer (standalone version).

```cpp
void make_rotation_matrix(
raft::resources const& res,
raft::device_matrix_view<float, uint32_t, raft::row_major> rotation_matrix,
bool force_random_rotation);
```

This standalone helper generates a rotation matrix without requiring an index object. Users can call this to prepare a rotation matrix before building from precomputed data.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `rotation_matrix` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | Output buffer [rot_dim, dim] for the rotation matrix |
| `force_random_rotation` | in | `bool` | If false and rot_dim == dim, creates identity matrix. If true or rot_dim != dim, creates random orthogonal matrix. |

**Returns**

`void`

<a id="neighbors-ivf-pq-helpers-resize-list"></a>
### neighbors::ivf_pq::helpers::resize_list

Resize an IVF-PQ list with flat layout.

```cpp
void resize_list(raft::resources const& res,
std::shared_ptr<list_data_base<int64_t, uint32_t>>& orig_list,
const list_spec_flat<uint32_t, int64_t>& spec,
uint32_t new_used_size,
uint32_t old_used_size);
```

This helper resizes an IVF list that uses the flat (non-interleaved) PQ code layout. If the new size exceeds the current capacity, a new list is allocated and existing data is copied. The function handles the type casting internally.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `orig_list` | inout | `std::shared_ptr<list_data_base<int64_t, uint32_t>>&` | the list to resize (may be replaced with a new allocation) |
| `spec` | in | [`const list_spec_flat<uint32_t, int64_t>&`](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-list-spec-flat) | the list specification containing pq_bits, pq_dim, and allocation settings |
| `new_used_size` | in | `uint32_t` | the new size of the list (number of vectors) |
| `old_used_size` | in | `uint32_t` | the current size of the list (data up to this size is preserved) |

**Returns**

`void`

**Additional overload:** `neighbors::ivf_pq::helpers::resize_list`

Resize an IVF-PQ list with interleaved layout.

```cpp
void resize_list(raft::resources const& res,
std::shared_ptr<list_data_base<int64_t, uint32_t>>& orig_list,
const list_spec_interleaved<uint32_t, int64_t>& spec,
uint32_t new_used_size,
uint32_t old_used_size);
```

This helper resizes an IVF list that uses the interleaved PQ code layout (default). If the new size exceeds the current capacity, a new list is allocated and existing data is copied. The function handles the type casting internally.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `orig_list` | inout | `std::shared_ptr<list_data_base<int64_t, uint32_t>>&` | the list to resize (may be replaced with a new allocation) |
| `spec` | in | `const list_spec_interleaved<uint32_t, int64_t>&` | the list specification containing pq_bits, pq_dim, and allocation settings |
| `new_used_size` | in | `uint32_t` | the new size of the list (number of vectors) |
| `old_used_size` | in | `uint32_t` | the current size of the list (data up to this size is preserved) |

**Returns**

`void`
