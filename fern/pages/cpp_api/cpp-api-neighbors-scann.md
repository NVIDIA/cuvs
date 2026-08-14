---
slug: api-reference/cpp-api-neighbors-scann
---

# Scann

_Source header: `cuvs/neighbors/scann.hpp`_

## ScaNN index build parameters

<a id="neighbors-experimental-scann-index-params"></a>
### neighbors::experimental::scann::index_params

ANN parameters used by ScaNN to build index

```cpp
struct index_params : cuvs::neighbors::index_params {
  uint32_t n_leaves;
  int64_t kmeans_n_rows_train;
  uint32_t kmeans_n_iters;
  float partitioning_eta;
  float soar_lambda;
  uint32_t pq_dim;
  uint32_t pq_bits;
  int64_t pq_n_rows_train;
  uint32_t pq_train_iters;
  bool reordering_bf16;
  float reordering_noise_shaping_threshold;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_leaves` | `uint32_t` | the number of leaves in the tree * |
| `kmeans_n_rows_train` | `int64_t` | the number of rows for training the tree structures * |
| `kmeans_n_iters` | `uint32_t` | the max number of iterations for training the tree structure * |
| `partitioning_eta` | `float` | the value of eta for AVQ adjustment during partitioning * |
| `soar_lambda` | `float` | the value of lambda for SOAR spilling * |
| `pq_dim` | `uint32_t` | the dimension of pq subspaces (must divide dataset dimension)* |
| `pq_bits` | `uint32_t` | the number of bits for pq codes (must be 4 or 8, for 16 and 256 codes respectively) * |
| `pq_n_rows_train` | `int64_t` | the number of rows for PQ training (internally capped to 100k) * |
| `pq_train_iters` | `uint32_t` | the max number of iterations for PQ training * |
| `reordering_bf16` | `bool` | whether to apply bf16 quantization of dataset vectors * |
| `reordering_noise_shaping_threshold` | `float` | Threshold T for computing AVQ eta = (dim - 1) ( T^2 / \|\| x \|\|^2) / ( 1 - T^2 / \|\| x \|\|^2)<br /><br />When quantizing a vector x to x_q, AVQ minimizes the loss function L(x, x_q) = eta * \|\| r_para \|\|^2 + \|\| r_perp \|\|^2, where r = x - x_q, r_para = &lt;r, x&gt; * x / \|\| x \|\|^2, r_perp = r - r_para<br /><br />Compared to L2 loss, This produces an x_q which better approximates the dot product of a query vector with x<br /><br />If the threshold is NAN, AVQ is not performed during bfloat16 quant |

## ScaNN index type

<a id="neighbors-experimental-scann-index"></a>
### neighbors::experimental::scann::index

ScaNN index.

The index stores the dataset and the ScaNN graph in device memory.

```cpp
template <typename T, typename IdxT>
struct index;
```

<a id="neighbors-experimental-scann-index-metric"></a>
### neighbors::experimental::scann::index::metric

Distance metric used for clustering.

```cpp
[[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="neighbors-experimental-scann-index-size"></a>
### neighbors::experimental::scann::index::size

Total length of the index (number of vectors).

```cpp
IdxT size() const noexcept;
```

**Returns**

`IdxT`

<a id="neighbors-experimental-scann-index-dim"></a>
### neighbors::experimental::scann::index::dim

Dimensionality of the data.

```cpp
[[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

## ScaNN index build functions

<a id="neighbors-experimental-scann-build"></a>
### neighbors::experimental::scann::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::experimental::scann::index_params& params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::experimental::scann::index<float, int64_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const cuvs::neighbors::experimental::scann::index_params&`](/api-reference/cpp-api-neighbors-scann#neighbors-experimental-scann-index-params) |  |
| `dataset` |  | `raft::device_matrix_view<const float, int64_t, raft::row_major>` |  |

**Returns**

[`cuvs::neighbors::experimental::scann::index<float, int64_t>`](/api-reference/cpp-api-neighbors-scann#neighbors-experimental-scann-index)

<a id="neighbors-experimental-scann-serialize"></a>
### neighbors::experimental::scann::serialize

Save the index to files in a directory

```cpp
void serialize(raft::resources const& handle,
const std::string& file_prefix,
const cuvs::neighbors::experimental::scann::index<float, int64_t>& index);
```

This serializes the index into a list of files for integration into OSS ScaNN for use with search

NOTE: the implementation of ScaNN index build is EXPERIMENTAL and currently not subject to comprehensive, automated testing. Accuracy and performance are not guaranteed, and could diverge without warning.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `file_prefix` |  | `const std::string&` |  |
| `index` |  | [`const cuvs::neighbors::experimental::scann::index<float, int64_t>&`](/api-reference/cpp-api-neighbors-scann#neighbors-experimental-scann-index) |  |

**Returns**

`void`

## ScaNN serialize functions

**Additional overload:** `neighbors::experimental::scann::serialize`

Save the index to files in a directory

```cpp
void serialize(raft::resources const& handle,
const std::string& file_prefix,
const cuvs::neighbors::experimental::scann::index<float, int64_t>& index);
```

This serializes the index into a list of files for integration into OSS ScaNN for use with search

NOTE: the implementation of ScaNN index build is EXPERIMENTAL and currently not subject to comprehensive, automated testing. Accuracy and performance are not guaranteed, and could diverge without warning.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `file_prefix` |  | `const std::string&` |  |
| `index` |  | [`const cuvs::neighbors::experimental::scann::index<float, int64_t>&`](/api-reference/cpp-api-neighbors-scann#neighbors-experimental-scann-index) |  |

**Returns**

`void`
