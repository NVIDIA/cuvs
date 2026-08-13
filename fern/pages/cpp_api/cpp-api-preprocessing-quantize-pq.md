---
slug: api-reference/cpp-api-preprocessing-quantize-pq
---

# PQ

_Source header: `cuvs/preprocessing/quantize/pq.hpp`_

## Product Quantizer utilities

<a id="kmeans-params-variant"></a>
### kmeans_params_variant

Alias for the variant holding either balanced or regular k-means parameters.

```cpp
using kmeans_params_variant =
std::variant<cuvs::cluster::kmeans::balanced_params, cuvs::cluster::kmeans::params>;
```

<a id="preprocessing-quantize-pq-params"></a>
### preprocessing::quantize::pq::params

Product Quantizer parameters.

```cpp
struct params {
  uint32_t pq_bits;
  uint32_t pq_dim;
  bool use_subspaces;
  bool use_vq;
  uint32_t vq_n_centers;
  kmeans_params_variant kmeans_params;
  uint32_t max_train_points_per_pq_code;
  uint32_t max_train_points_per_vq_cluster;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ.<br /><br />Possible value range: [4-16].<br /><br />Hint: the smaller the 'pq_bits', the smaller the index size and the faster the fit/transform time, but the lower the recall. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, dim / 4 is used as default.<br /><br />TODO: at the moment `dim` must be a multiple `pq_dim`. |
| `use_subspaces` | `bool` | Whether to use subspaces for product quantization (PQ). When true, one PQ codebook is used for each subspace. Otherwise, a single PQ codebook is used. |
| `use_vq` | `bool` | Whether to use Vector Quantization (KMeans) before product quantization (PQ). When true, VQ is used and PQ is trained on the residuals. |
| `vq_n_centers` | `uint32_t` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". When zero, an optimal value is selected using a heuristic. (sqrt(n_rows)) |
| `kmeans_params` | [`kmeans_params_variant`](/api-reference/cpp-api-preprocessing-quantize-pq#kmeans-params-variant) | K-means parameters for PQ codebook training.<br /><br />Set to cuvs::cluster::kmeans::balanced_params for balanced k-means (default), or cuvs::cluster::kmeans::params for regular k-means. The active variant type selects the algorithm; balanced k-means tends to be faster for PQ training where cluster sizes are approximately equal. Only L2Expanded metric is supported. The number of clusters is always set to 1 &lt;&lt; pq_bits. |
| `max_train_points_per_pq_code` | `uint32_t` | The max number of data points to use per PQ code during PQ codebook training. Using more data points per PQ code may increase the quality of PQ codebook but may also increase the build time. We will use `pq_n_centers * max_train_points_per_pq_code` training points to train each PQ codebook. |
| `max_train_points_per_vq_cluster` | `uint32_t` | The max number of data points to use per VQ cluster during training. |

<a id="preprocessing-quantize-pq-params-params"></a>
### preprocessing::quantize::pq::params::params

Simplified constructor that will build an appropriate kmeans params object.

```cpp
params(uint32_t pq_bits,
uint32_t pq_dim,
bool use_subspaces,
bool use_vq,
uint32_t vq_n_centers,
uint32_t kmeans_n_iters,
cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
uint32_t max_train_points_per_pq_code    = 256,
uint32_t max_train_points_per_vq_cluster = 1024);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `pq_bits` |  | `uint32_t` |  |
| `pq_dim` |  | `uint32_t` |  |
| `use_subspaces` |  | `bool` |  |
| `use_vq` |  | `bool` |  |
| `vq_n_centers` |  | `uint32_t` |  |
| `kmeans_n_iters` |  | `uint32_t` |  |
| `pq_kmeans_type` |  | [`cuvs::cluster::kmeans::kmeans_type`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-kmeans-type) | Default: `cuvs::cluster::kmeans::kmeans_type::KMeansBalanced`. |
| `max_train_points_per_pq_code` |  | `uint32_t` | Default: `256`. |
| `max_train_points_per_vq_cluster` |  | `uint32_t` | Default: `1024`. |

**Returns**

`void`

<a id="preprocessing-quantize-pq-quantizer"></a>
### preprocessing::quantize::pq::quantizer

Defines and stores VPQ codebooks upon training

```cpp
template <typename T>
struct quantizer {
  params params_quantizer;
  cuvs::neighbors::device_vpq_dataset<T, int64_t> vpq_codebooks;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `params_quantizer` | [`params`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-params) | Parameters used to build this quantizer. |
| `vpq_codebooks` | [`cuvs::neighbors::device_vpq_dataset<T, int64_t>`](/api-reference/cpp-api-neighbors-common#neighbors-vpq-dataset) | VPQ codebooks produced during training. |

<a id="preprocessing-quantize-pq-build"></a>
### preprocessing::quantize::pq::build

Initializes a product quantizer to be used later for quantizing the dataset.

```cpp
quantizer<float> build(raft::resources const& res,
const params params,
raft::device_matrix_view<const float, int64_t> dataset);
```

The use of a pool memory resource is recommended for more consistent training performance.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `params` | in | [`const params`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-params) | configure product quantizer, e.g. quantile |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device or host |

**Returns**

[`quantizer<float>`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-quantizer)

**Additional overload:** `preprocessing::quantize::pq::build`

```cpp
quantizer<float> build(raft::resources const& res,
const params params,
raft::host_matrix_view<const float, int64_t> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`const params`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-params) |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t>` |  |

**Returns**

[`quantizer<float>`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-quantizer)

<a id="preprocessing-quantize-pq-transform"></a>
### preprocessing::quantize::pq::transform

Applies quantization transform to given dataset

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quant,
raft::device_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> codes_out,
std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quant` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-quantizer) | a product quantizer |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t>` | a row-major matrix view on device or host |
| `codes_out` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a row-major matrix view on device containing the PQ codes |
| `vq_labels` | out | `std::optional<raft::device_vector_view<uint32_t, int64_t>>` | a vector view on device containing the VQ labels when VQ is used, optional<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `preprocessing::quantize::pq::transform`

```cpp
void transform(raft::resources const& res,
const quantizer<float>& quant,
raft::host_matrix_view<const float, int64_t> dataset,
raft::device_matrix_view<uint8_t, int64_t> codes_out,
std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `quant` |  | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-quantizer) |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t>` |  |
| `codes_out` |  | `raft::device_matrix_view<uint8_t, int64_t>` |  |
| `vq_labels` |  | `std::optional<raft::device_vector_view<uint32_t, int64_t>>` | Default: `std::nullopt`. |

**Returns**

`void`

<a id="preprocessing-quantize-pq-get-quantized-dim"></a>
### preprocessing::quantize::pq::get_quantized_dim

Get the dimension of the quantized dataset (in bytes)

```cpp
inline int64_t get_quantized_dim(const params& config);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `config` | in | [`const params&`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-params) | product quantizer parameters |

**Returns**

`inline int64_t`

the dimension of the quantized dataset

<a id="preprocessing-quantize-pq-inverse-transform"></a>
### preprocessing::quantize::pq::inverse_transform

Applies inverse quantization transform to given dataset

```cpp
void inverse_transform(
raft::resources const& res,
const quantizer<float>& quant,
raft::device_matrix_view<const uint8_t, int64_t> pq_codes,
raft::device_matrix_view<float, int64_t> out,
std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `quant` | in | [`const quantizer<float>&`](/api-reference/cpp-api-preprocessing-quantize-pq#preprocessing-quantize-pq-quantizer) | a product quantizer |
| `pq_codes` | in | `raft::device_matrix_view<const uint8_t, int64_t>` | a row-major matrix view on device containing the PQ codes |
| `out` | out | `raft::device_matrix_view<float, int64_t>` | a row-major matrix view on device |
| `vq_labels` | in | `std::optional<raft::device_vector_view<const uint32_t, int64_t>>` | a vector view on device containing the VQ labels when VQ is used, optional<br />Default: `std::nullopt`. |

**Returns**

`void`
