---
slug: api-reference/cpp-api-neighbors-hnsw
---

# HNSW

_Source header: `cuvs/neighbors/hnsw.hpp`_

## hnswlib index wrapper params

<a id="neighbors-hnsw-hnswhierarchy"></a>
### neighbors::hnsw::HnswHierarchy

Hierarchy for HNSW index when converting from CAGRA index

NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only index. When the value is `CPU`, a full index is built with a CPU-constructed hierarchy. When the value is `GPU`, a full index is built with a GPU-constructed hierarchy.

```cpp
enum class HnswHierarchy {
  NONE,
  CPU,
  GPU
};
```

**Values**

| Name | Value |
| --- | --- |
| `NONE` | `` |
| `CPU` | `` |
| `GPU` | `` |

<a id="neighbors-hnsw-hnswoutputformat"></a>
### neighbors::hnsw::HnswOutputFormat

Output artifact format for an HNSW index

`HNSWLIB` produces the standard hnswlib index format. `GRAPH_ONLY` stores the graph separately from vectors. The current implementation requires `HnswHierarchy::GPU` and disk-backed ACE. Load a graph-only artifact with the two-filename `deserialize` overload, which reads the vectors from a separate local dataset.

```cpp
enum class HnswOutputFormat {
  HNSWLIB,
  GRAPH_ONLY
};
```

**Values**

| Name | Value |
| --- | --- |
| `HNSWLIB` | `` |
| `GRAPH_ONLY` | `` |

<a id="neighbors-hnsw-index-params"></a>
### neighbors::hnsw::index_params

```cpp
struct index_params : cuvs::neighbors::index_params {
  HnswHierarchy hierarchy;
  HnswOutputFormat output_format;
  int ef_construction;
  int num_threads;
  size_t M;
  std::variant<std::monostate, graph_build_params::ace_params> graph_build_params;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `hierarchy` | [`HnswHierarchy`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-hnswhierarchy) | Hierarchy build type for HNSW index when converting from CAGRA index |
| `output_format` | [`HnswOutputFormat`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-hnswoutputformat) | Output artifact format. Graph-only output currently requires a GPU hierarchy and disk-backed ACE. |
| `ef_construction` | `int` | Size of the candidate list during hierarchy construction when hierarchy is `CPU` |
| `num_threads` | `int` | Number of host threads to use to construct hierarchy when hierarchy is `CPU` or `GPU`. When the value is 0, the number of threads is automatically determined to the maximum number of threads available.<br />NOTE: When hierarchy is `GPU`, while the majority of the work is done on the GPU, initialization of the HNSW index itself and some other work is parallelized with the help of CPU threads. |
| `M` | `size_t` | HNSW M parameter: number of bi-directional links per node (used when building with ACE). |
| `graph_build_params` | `std::variant<std::monostate, graph_build_params::ace_params>` | Parameters to fine tune GPU graph building. By default we select the parameters based on dataset shape and HNSW build parameters. You can override these parameters to fine tune the graph building process as described in the CAGRA build docs.<br /><br />Set ace_params to configure ACE (Augmented Core Extraction) parameters for building a GPU-accelerated HNSW index. ACE enables building indexes for datasets too large to fit in GPU memory.<br /><br />When ACE writes to disk, `build_dir` may already exist, but ACE's named CAGRA artifacts and the selected HNSW output (`hnsw_index.bin` or `hnsw_index.cuvs`) must not already exist. Simultaneous builds must use different directories. The HNSW output is published only after it is fully serialized; however, the complete build is not transactional: if HNSW conversion fails after CAGRA succeeds, the completed CAGRA artifacts remain in the directory. |

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
| `arg1` |  | [`"Use cagra::index_params::from_hnsw_params instead"`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) |  |

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
index(int dim,
cuvs::distance::DistanceType metric,
HnswHierarchy hierarchy        = HnswHierarchy::NONE,
HnswOutputFormat output_format = HnswOutputFormat::HNSWLIB);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `hierarchy` | in | [`HnswHierarchy`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-hnswhierarchy) | hierarchy used for upper HNSW layers<br />Default: `HnswHierarchy::NONE`. |
| `output_format` | in | [`HnswOutputFormat`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-hnswoutputformat) | output artifact format<br />Default: `HnswOutputFormat::HNSWLIB`. |

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

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality.  We have additional options that can be used to fine tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is important to set `ace_params.build_dir` to a fast disk with sufficient storage size.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters including ACE configuration |
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

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality.  We have additional options that can be used to fine tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is important to set `ace_params.build_dir` to a fast disk with sufficient storage size.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters including ACE configuration |
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

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality.  We have additional options that can be used to fine tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is important to set `ace_params.build_dir` to a fast disk with sufficient storage size.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters including ACE configuration |
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

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality.  We have additional options that can be used to fine tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is important to set `ace_params.build_dir` to a fast disk with sufficient storage size.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters including ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<int8_t>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

## Load CAGRA index as hnswlib index

<a id="neighbors-hnsw-from-cagra"></a>
### neighbors::hnsw::from_cagra

Construct an hnswlib index from a CAGRA index NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.
3. `GPU`: The hierarchy is constructed on the GPU. When `output_format` is `GRAPH_ONLY`, the GPU-built hierarchy is stored as graph links only. Reload it with the two-filename `deserialize` overload so vectors can be reconstructed from the local dataset.

```cpp
std::unique_ptr<index<float>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::device_padded_index<float, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters |
| `cagra_index` | in | `const cuvs::neighbors::cagra::device_padded_index<float, uint32_t>&` | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU`<br /><br />Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<float>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.
3. `GPU`: The hierarchy is constructed on the GPU. When `output_format` is `GRAPH_ONLY`, the GPU-built hierarchy is stored as graph links only. Reload it with the two-filename `deserialize` overload so vectors can be reconstructed from the local dataset.

```cpp
std::unique_ptr<index<half>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::device_padded_index<half, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters |
| `cagra_index` | in | `const cuvs::neighbors::cagra::device_padded_index<half, uint32_t>&` | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU`<br /><br />Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<half>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.
3. `GPU`: The hierarchy is constructed on the GPU. When `output_format` is `GRAPH_ONLY`, the GPU-built hierarchy is stored as graph links only. Reload it with the two-filename `deserialize` overload so vectors can be reconstructed from the local dataset.

```cpp
std::unique_ptr<index<uint8_t>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters |
| `cagra_index` | in | `const cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>&` | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU`<br /><br />Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<uint8_t>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.
3. `GPU`: The hierarchy is constructed on the GPU. When `output_format` is `GRAPH_ONLY`, the GPU-built hierarchy is stored as graph links only. Reload it with the two-filename `deserialize` overload so vectors can be reconstructed from the local dataset.

```cpp
std::unique_ptr<index<int8_t>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters |
| `cagra_index` | in | `const cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>&` | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU`<br /><br />Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<int8_t>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a device-standard CAGRA index.

```cpp
std::unique_ptr<index<float>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::device_standard_index<float, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

When the index has an attached device dataset view, `dataset` may be omitted. Otherwise pass a host matrix with the vectors (same contract as `device_padded_index`).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) |  |
| `cagra_index` |  | `const cuvs::neighbors::cagra::device_standard_index<float, uint32_t>&` |  |
| `dataset` |  | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<float>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a host-built CAGRA index. Requires `dataset` for in-memory indices — host builds do not store vectors in the index.

```cpp
std::unique_ptr<index<float>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::host_padded_index<float, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) |  |
| `cagra_index` |  | `const cuvs::neighbors::cagra::host_padded_index<float, uint32_t>&` |  |
| `dataset` |  | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<float>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

**Additional overload:** `neighbors::hnsw::from_cagra`

Construct an hnswlib index from a host-built CAGRA index (standard dataset layout). Requires `dataset` for in-memory indices — host builds do not store vectors in the index.

```cpp
std::unique_ptr<index<float>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::host_standard_index<float, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) |  |
| `cagra_index` |  | `const cuvs::neighbors::cagra::host_standard_index<float, uint32_t>&` |  |
| `dataset` |  | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<float>>`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index)

## Extend HNSW index with additional vectors

<a id="neighbors-hnsw-extend"></a>
### neighbors::hnsw::extend

Add new vectors to an HNSW index NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU` when converting from a CAGRA index.

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

Add new vectors to an HNSW index NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU` when converting from a CAGRA index.

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

Add new vectors to an HNSW index NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU` when converting from a CAGRA index.

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

Add new vectors to an HNSW index NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU` when converting from a CAGRA index.

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

Search HNSW index constructed from a CAGRA index NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib. When `output_format` is `GRAPH_ONLY`, search uses the in-memory index reconstructed from the graph-only artifact produced by the two-filename `deserialize` overload.

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
| `idx` | in | [`const index<float>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |
| `queries` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::search`

Search HNSW index constructed from a CAGRA index NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib. When `output_format` is `GRAPH_ONLY`, search uses the in-memory index reconstructed from the graph-only artifact produced by the two-filename `deserialize` overload.

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
| `idx` | in | [`const index<half>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |
| `queries` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::search`

Search HNSWindex constructed from a CAGRA index NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib. When `output_format` is `GRAPH_ONLY`, search uses the in-memory index reconstructed from the graph-only artifact produced by the two-filename `deserialize` overload.

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
| `idx` | in | [`const index<uint8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |
| `queries` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::search`

Search HNSW index constructed from a CAGRA index NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib. When `output_format` is `GRAPH_ONLY`, search uses the in-memory index reconstructed from the graph-only artifact produced by the two-filename `deserialize` overload.

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
| `idx` | in | [`const index<int8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |
| `queries` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

## Serialize and deserialize HNSW indexes

<a id="neighbors-hnsw-serialize"></a>
### neighbors::hnsw::serialize

Serialize the HNSW index to file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library. When `output_format` is `GRAPH_ONLY`, the saved artifact stores the graph only. Load it with the two-filename `deserialize` overload and a local dataset.

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<float>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the serialized HNSW output |
| `idx` | in | [`const index<float>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::serialize`

Serialize the HNSW index to file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library. When `output_format` is `GRAPH_ONLY`, the saved artifact stores the graph only. Load it with the two-filename `deserialize` overload and a local dataset.

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<half>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the serialized HNSW output |
| `idx` | in | [`const index<half>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::serialize`

Serialize the HNSW index to file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library. When `output_format` is `GRAPH_ONLY`, the saved artifact stores the graph only. Load it with the two-filename `deserialize` overload and a local dataset.

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<uint8_t>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the serialized HNSW output |
| `idx` | in | [`const index<uint8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::serialize`

Serialize the HNSW index to file NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library. When `output_format` is `GRAPH_ONLY`, the saved artifact stores the graph only. Load it with the two-filename `deserialize` overload and a local dataset.

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<int8_t>& idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the serialized HNSW output |
| `idx` | in | [`const index<int8_t>&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | HNSW index |

**Returns**

`void`

<a id="neighbors-hnsw-deserialize"></a>
### neighbors::hnsw::deserialize

Deserialize an HNSWLIB index NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library. This overload loads `HNSWLIB` artifacts. Use the two-filename overload for graph-only artifacts.

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
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the HNSWLIB artifact |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `index` | out | [`index<float>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

Deserialize an HNSWLIB index NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library. This overload loads `HNSWLIB` artifacts. Use the two-filename overload for graph-only artifacts.

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
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the HNSWLIB artifact |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `index` | out | [`index<half>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

Deserialize an HNSWLIB index NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library. This overload loads `HNSWLIB` artifacts. Use the two-filename overload for graph-only artifacts.

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
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the HNSWLIB artifact |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `index` | out | [`index<uint8_t>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

Deserialize an HNSWLIB index NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library. This overload loads `HNSWLIB` artifacts. Use the two-filename overload for graph-only artifacts.

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
| `params` | in | [`const index_params&`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index-params) | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the HNSWLIB artifact |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `index` | out | [`index<int8_t>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

Deserialize a graph-only HNSW artifact and attach its dataset

```cpp
void deserialize(raft::resources const& res,
const std::string& graph_filename,
const std::string& dataset_filename,
index<float>** index);
```

The graph artifact supplies the index dimensions, metric, and construction metadata. The attached dataset must have the recorded shape, but its element type may differ from the type used to construct the graph. The output pointer selects the attached dataset type. The loader accepts row-major `.npy` files and ANN benchmark binary files with a `[uint32 rows, uint32 cols]` header. Binary extensions must match the output index type: `.fbin` for `float`, `.f16bin` or `.fp16.fbin` for `half`, `.u8bin` for `uint8_t`, and `.i8bin` for `int8_t`.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `graph_filename` | in | `const std::string&` | path to the graph-only HNSW artifact |
| `dataset_filename` | in | `const std::string&` | path to the local dataset |
| `index` | out | [`index<float>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) | reconstructed HNSW index |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

```cpp
void deserialize(raft::resources const& res,
const std::string& graph_filename,
const std::string& dataset_filename,
index<half>** index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `graph_filename` |  | `const std::string&` |  |
| `dataset_filename` |  | `const std::string&` |  |
| `index` |  | [`index<half>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

```cpp
void deserialize(raft::resources const& res,
const std::string& graph_filename,
const std::string& dataset_filename,
index<uint8_t>** index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `graph_filename` |  | `const std::string&` |  |
| `dataset_filename` |  | `const std::string&` |  |
| `index` |  | [`index<uint8_t>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) |  |

**Returns**

`void`

**Additional overload:** `neighbors::hnsw::deserialize`

```cpp
void deserialize(raft::resources const& res,
const std::string& graph_filename,
const std::string& dataset_filename,
index<int8_t>** index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `graph_filename` |  | `const std::string&` |  |
| `dataset_filename` |  | `const std::string&` |  |
| `index` |  | [`index<int8_t>**`](/api-reference/cpp-api-neighbors-hnsw#neighbors-hnsw-index) |  |

**Returns**

`void`
