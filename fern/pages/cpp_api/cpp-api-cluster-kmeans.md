---
slug: api-reference/cpp-api-cluster-kmeans
---

# K-Means

_Source header: `cuvs/cluster/kmeans.hpp`_

## Types

<a id="cluster-kmeans-base-params"></a>
### cluster::kmeans::base_params

Base structure for parameters that are common to all k-means algorithms

```cpp
struct base_params {
  cuvs::distance::DistanceType metric;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | Metric to use for distance computation. The supported metrics can vary per algorithm. |

## k-means hyperparameters

<a id="cluster-kmeans-params"></a>
### cluster::kmeans::params

Simple object to specify hyper-parameters to the kmeans algorithm.

```cpp
struct params : base_params {
  int n_clusters;
  InitMethod init;
  int max_iter;
  double tol;
  rapids_logger::level_enum verbosity;
  raft::random::RngState rng_state;
  int n_init;
  double oversampling_factor;
  int batch_samples;
  int batch_centroids;
  int64_t init_size;
  int64_t device_buffer_samples;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_clusters` | `int` | The number of clusters to form as well as the number of centroids to generate (default:8). |
| `init` | `InitMethod` | Method for initialization, defaults to k-means++:<br />- InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm to select the initial cluster centers.<br />- InitMethod::Random (random): Choose 'n_clusters' observations (rows) at random from the input data for the initial centroids.<br />- InitMethod::Array (ndarray): Use 'centroids' as initial cluster centers. |
| `max_iter` | `int` | Maximum number of iterations of the k-means algorithm for a single run. |
| `tol` | `double` | Relative tolerance with regards to inertia to declare convergence. |
| `verbosity` | `rapids_logger::level_enum` | verbosity level. |
| `rng_state` | `raft::random::RngState` | Seed to the random number generator. |
| `n_init` | `int` | Number of instance k-means algorithm will be run with different seeds. |
| `oversampling_factor` | `double` | Oversampling factor for use in the k-means\|\| algorithm |
| `batch_samples` | `int` | batch_samples and batch_centroids are used to tile 1NN computation which is useful to optimize/control the memory footprint<br />Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0 then don't tile the centroids<br /><br />NB: These parameters are unrelated to device_buffer_samples, which controls how many samples to transfer from host to device per batch when processing out-of-core data. |
| `batch_centroids` | `int` | if 0 then batch_centroids = n_clusters |
| `init_size` | `int64_t` | Number of samples to randomly draw for the KMeansPlusPlus initialization step. A random subset of this size is used for centroid seeding.<br /><br />Only applies when dataset is on host; for device data the full dataset is always used for seeding and this parameter is ignored.<br /><br />When set to 0 (default) with host data uses `min(3 * n_clusters, n_samples)` as a default.<br /><br />In Batched multi-GPU host-data fits, the effective KMeansPlusPlus initialization sample is materialized on device on every rank. Every rank must have enough GPU memory for this sample, and rank 0 must also have enough GPU memory for the seeding workspace.<br /><br />Default: 0. |
| `device_buffer_samples` | `int64_t` | Number of samples to process per GPU batch when fitting with host data. When set to 0, defaults to n_samples (process all at once). Only used by the batched (host-data) code path and ignored by device-data overloads.<br /><br />In multi-GPU mode, this is a per-rank batch size. Each rank processes up to this many local samples per batch, clamped to that rank's local sample count.<br />Default: 0 (process all data at once). |

<a id="cluster-kmeans-balanced-params"></a>
### cluster::kmeans::balanced_params

Simple object to specify hyper-parameters to the balanced k-means algorithm.

The following metrics are currently supported in k-means balanced:

- CosineExpanded
- InnerProduct
- L2Expanded
- L2SqrtExpanded

```cpp
struct balanced_params : base_params {
  uint32_t n_iters;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_iters` | `uint32_t` | Number of training iterations |

<a id="cluster-kmeans-kmeans-type"></a>
### cluster::kmeans::kmeans_type

Type of k-means algorithm.

```cpp
enum class kmeans_type {
  KMeans = 0,
  KMeansBalanced = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `KMeans` | `0` |
| `KMeansBalanced` | `1` |

## k-means clustering APIs

<a id="cluster-kmeans-fit"></a>
### cluster::kmeans::fit

Find clusters with k-means algorithm using batched processing of host data.

```cpp
void fit(raft::resources const& handle,
const cuvs::cluster::kmeans::params& params,
raft::host_matrix_view<const float, int64_t> X,
std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
raft::device_matrix_view<float, int64_t> centroids,
raft::host_scalar_view<float> inertia,
raft::host_scalar_view<int64_t> n_iter);
```

TODO: Evaluate replacing the extent type with int64_t. Reference issue: https://github.com/rapidsai/cuvs/issues/1961

This overload supports out-of-core computation where the dataset resides on the host. Data is processed in GPU-sized batches, streaming from host to device. The batch size is controlled by params.device_buffer_samples. In multi-GPU mode, this is a per-rank batch size.

Multi-GPU dispatch is selected automatically based on the handle state:

- If `raft::resource::is_multi_gpu(handle)` (cuVS SNMG): the full dataset X is split across GPUs internally with an OpenMP parallel region and NCCL.
- If `raft::resource::comms_initialized(handle)` (Dask/Ray/MPI): X is treated as this worker's partition, and RAFT communicators are used for collectives.
- Otherwise: single-GPU batched k-means.

With `params.init == InitMethod::KMeansPlusPlus` in multi-GPU mode, the effective initialization sample must fit in GPU memory on every rank because it is materialized on every device. Rank 0 must also have enough GPU memory for the seeding workspace before centroids are broadcast.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. When a multi-GPU resource is attached, multi-GPU dispatch is used automatically. |
| `params` | in | [`const cuvs::cluster::kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. Batch size is read from params.device_buffer_samples. |
| `X` | in | `raft::host_matrix_view<const float, int64_t>` | Training instances on HOST memory. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::host_vector_view<const float, int64_t>>` | Optional weights for each observation in X (on host). [len = n_samples] |
| `centroids` | inout | `raft::device_matrix_view<float, int64_t>` | [in] When init is InitMethod::Array, use centroids as the initial cluster centers. [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `raft::host_scalar_view<float>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int64_t>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find clusters with k-means algorithm using batched processing of host data.

```cpp
void fit(raft::resources const& handle,
const cuvs::cluster::kmeans::params& params,
raft::host_matrix_view<const double, int64_t> X,
std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
raft::device_matrix_view<double, int64_t> centroids,
raft::host_scalar_view<double> inertia,
raft::host_scalar_view<int64_t> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const cuvs::cluster::kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) |  |
| `X` |  | `raft::host_matrix_view<const double, int64_t>` |  |
| `sample_weight` |  | `std::optional<raft::host_vector_view<const double, int64_t>>` |  |
| `centroids` |  | `raft::device_matrix_view<double, int64_t>` |  |
| `inertia` |  | `raft::host_scalar_view<double>` |  |
| `n_iter` |  | `raft::host_scalar_view<int64_t>` |  |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find clusters with k-means algorithm. Initial centroids are chosen with k-means++ algorithm. Empty clusters are reinitialized by choosing new centroids with k-means++ algorithm.

```cpp
void fit(raft::resources const& handle,
const cuvs::cluster::kmeans::params& params,
raft::device_matrix_view<const float, int> X,
std::optional<raft::device_vector_view<const float, int>> sample_weight,
raft::device_matrix_view<float, int> centroids,
raft::host_scalar_view<float> inertia,
raft::host_scalar_view<int> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const cuvs::cluster::kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const float, int>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `raft::device_matrix_view<float, int>` | [in] When init is InitMethod::Array, use centroids as the initial cluster centers. [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `raft::host_scalar_view<float>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find clusters with k-means algorithm. Initial centroids are chosen with k-means++ algorithm. Empty clusters are reinitialized by choosing new centroids with k-means++ algorithm.

```cpp
void fit(raft::resources const& handle,
const cuvs::cluster::kmeans::params& params,
raft::device_matrix_view<const float, int64_t> X,
std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
raft::device_matrix_view<float, int64_t> centroids,
raft::host_scalar_view<float> inertia,
raft::host_scalar_view<int64_t> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const cuvs::cluster::kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const float, int64_t>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `raft::device_matrix_view<float, int64_t>` | [in] When init is InitMethod::Array, use centroids as the initial cluster centers. [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `raft::host_scalar_view<float>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int64_t>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find clusters with k-means algorithm. Initial centroids are chosen with k-means++ algorithm. Empty clusters are reinitialized by choosing new centroids with k-means++ algorithm.

```cpp
void fit(raft::resources const& handle,
const cuvs::cluster::kmeans::params& params,
raft::device_matrix_view<const double, int> X,
std::optional<raft::device_vector_view<const double, int>> sample_weight,
raft::device_matrix_view<double, int> centroids,
raft::host_scalar_view<double> inertia,
raft::host_scalar_view<int> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const cuvs::cluster::kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const double, int>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const double, int>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `raft::device_matrix_view<double, int>` | [in] When init is InitMethod::Array, use centroids as the initial cluster centers. [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `raft::host_scalar_view<double>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find clusters with k-means algorithm. Initial centroids are chosen with k-means++ algorithm. Empty clusters are reinitialized by choosing new centroids with k-means++ algorithm.

```cpp
void fit(raft::resources const& handle,
const cuvs::cluster::kmeans::params& params,
raft::device_matrix_view<const double, int64_t> X,
std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
raft::device_matrix_view<double, int64_t> centroids,
raft::host_scalar_view<double> inertia,
raft::host_scalar_view<int64_t> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const cuvs::cluster::kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const double, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const double, int64_t>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `raft::device_matrix_view<double, int64_t>` | [in] When init is InitMethod::Array, use centroids as the initial cluster centers. [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `raft::host_scalar_view<double>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int64_t>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find clusters with k-means algorithm. Initial centroids are chosen with k-means++ algorithm. Empty clusters are reinitialized by choosing new centroids with k-means++ algorithm.

```cpp
void fit(raft::resources const& handle,
const cuvs::cluster::kmeans::params& params,
raft::device_matrix_view<const int8_t, int> X,
std::optional<raft::device_vector_view<const int8_t, int>> sample_weight,
raft::device_matrix_view<int8_t, int> centroids,
raft::host_scalar_view<int8_t> inertia,
raft::host_scalar_view<int> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const cuvs::cluster::kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const int8_t, int>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const int8_t, int>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `raft::device_matrix_view<int8_t, int>` | [in] When init is InitMethod::Array, use centroids as the initial cluster centers. [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `raft::host_scalar_view<int8_t>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find balanced clusters with k-means algorithm.

```cpp
void fit(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const float, int64_t> X,
raft::device_matrix_view<float, int64_t> centroids,
std::optional<raft::host_scalar_view<float>> inertia = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | out | `raft::device_matrix_view<float, int64_t>` | [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `std::optional<raft::host_scalar_view<float>>` | Sum of squared distances of samples to their closest cluster center.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find balanced clusters with k-means algorithm.

```cpp
void fit(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const int8_t, int64_t> X,
raft::device_matrix_view<float, int64_t> centroids,
std::optional<raft::host_scalar_view<float>> inertia = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const int8_t, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | inout | `raft::device_matrix_view<float, int64_t>` | [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `std::optional<raft::host_scalar_view<float>>` | Sum of squared distances of samples to their closest cluster center.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find balanced clusters with k-means algorithm.

```cpp
void fit(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const half, int64_t> X,
raft::device_matrix_view<float, int64_t> centroids,
std::optional<raft::host_scalar_view<float>> inertia = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const half, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | inout | `raft::device_matrix_view<float, int64_t>` | [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `std::optional<raft::host_scalar_view<float>>` | Sum of squared distances of samples to their closest cluster center.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit`

Find balanced clusters with k-means algorithm.

```cpp
void fit(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const uint8_t, int64_t> X,
raft::device_matrix_view<float, int64_t> centroids,
std::optional<raft::host_scalar_view<float>> inertia = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const uint8_t, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | inout | `raft::device_matrix_view<float, int64_t>` | [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `inertia` | out | `std::optional<raft::host_scalar_view<float>>` | Sum of squared distances of samples to their closest cluster center.<br />Default: `std::nullopt`. |

**Returns**

`void`

<a id="cluster-kmeans-predict"></a>
### cluster::kmeans::predict

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const float, int> X,
std::optional<raft::device_vector_view<const float, int>> sample_weight,
raft::device_matrix_view<const float, int> centroids,
raft::device_vector_view<int, int> labels,
bool normalize_weight,
raft::host_scalar_view<float> inertia);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int>` | New data to predict. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const float, int>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | in | `raft::device_matrix_view<const float, int>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int, int>` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `normalize_weight` | in | `bool` | True if the weights should be normalized |
| `inertia` | out | `raft::host_scalar_view<float>` | Sum of squared distances of samples to their closest cluster center. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const float, int64_t> X,
std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_vector_view<int64_t, int64_t> labels,
bool normalize_weight,
raft::host_scalar_view<float> inertia);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | New data to predict. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const float, int64_t>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int64_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `normalize_weight` | in | `bool` | True if the weights should be normalized |
| `inertia` | out | `raft::host_scalar_view<float>` | Sum of squared distances of samples to their closest cluster center. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const double, int> X,
std::optional<raft::device_vector_view<const double, int>> sample_weight,
raft::device_matrix_view<const double, int> centroids,
raft::device_vector_view<int, int> labels,
bool normalize_weight,
raft::host_scalar_view<double> inertia);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const double, int>` | New data to predict. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const double, int>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | in | `raft::device_matrix_view<const double, int>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int, int>` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `normalize_weight` | in | `bool` | True if the weights should be normalized |
| `inertia` | out | `raft::host_scalar_view<double>` | Sum of squared distances of samples to their closest cluster center. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const double, int64_t> X,
std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
raft::device_matrix_view<const double, int64_t> centroids,
raft::device_vector_view<int64_t, int64_t> labels,
bool normalize_weight,
raft::host_scalar_view<double> inertia);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const double, int64_t>` | New data to predict. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const double, int64_t>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | in | `raft::device_matrix_view<const double, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int64_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `normalize_weight` | in | `bool` | True if the weights should be normalized |
| `inertia` | out | `raft::host_scalar_view<double>` | Sum of squared distances of samples to their closest cluster center. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const int8_t, int64_t> X,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_vector_view<uint32_t, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const int8_t, int64_t>` | New data to predict. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const int8_t, int64_t> X,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_vector_view<int, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const int8_t, int64_t>` | New data to predict. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const float, int64_t> X,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_vector_view<int, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | New data to predict. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const float, int64_t> X,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_vector_view<uint32_t, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | New data to predict. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const half, int64_t> X,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_vector_view<uint32_t, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const half, int64_t>` | New data to predict. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::predict`

Predict the closest cluster each sample in X belongs to.

```cpp
void predict(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const uint8_t, int64_t> X,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_vector_view<uint32_t, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const uint8_t, int64_t>` | New data to predict. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |

**Returns**

`void`

<a id="cluster-kmeans-fit-predict"></a>
### cluster::kmeans::fit_predict

Compute k-means clustering and predicts cluster index for each sample in the input.

```cpp
void fit_predict(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const float, int> X,
std::optional<raft::device_vector_view<const float, int>> sample_weight,
std::optional<raft::device_matrix_view<float, int>> centroids,
raft::device_vector_view<int, int> labels,
raft::host_scalar_view<float> inertia,
raft::host_scalar_view<int> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const float, int>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `std::optional<raft::device_matrix_view<float, int>>` | Optional [in] When init is InitMethod::Array, use centroids  as the initial cluster centers [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int, int>` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `inertia` | out | `raft::host_scalar_view<float>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit_predict`

Compute k-means clustering and predicts cluster index for each sample in the input.

```cpp
void fit_predict(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const float, int64_t> X,
std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
std::optional<raft::device_matrix_view<float, int64_t>> centroids,
raft::device_vector_view<int64_t, int64_t> labels,
raft::host_scalar_view<float> inertia,
raft::host_scalar_view<int64_t> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const float, int64_t>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `std::optional<raft::device_matrix_view<float, int64_t>>` | Optional [in] When init is InitMethod::Array, use centroids  as the initial cluster centers [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int64_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `inertia` | out | `raft::host_scalar_view<float>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int64_t>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit_predict`

Compute k-means clustering and predicts cluster index for each sample in the input.

```cpp
void fit_predict(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const double, int> X,
std::optional<raft::device_vector_view<const double, int>> sample_weight,
std::optional<raft::device_matrix_view<double, int>> centroids,
raft::device_vector_view<int, int> labels,
raft::host_scalar_view<double> inertia,
raft::host_scalar_view<int> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const double, int>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const double, int>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `std::optional<raft::device_matrix_view<double, int>>` | Optional [in] When init is InitMethod::Array, use centroids  as the initial cluster centers [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int, int>` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `inertia` | out | `raft::host_scalar_view<double>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit_predict`

Compute k-means clustering and predicts cluster index for each sample in the input.

```cpp
void fit_predict(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const double, int64_t> X,
std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
std::optional<raft::device_matrix_view<double, int64_t>> centroids,
raft::device_vector_view<int64_t, int64_t> labels,
raft::host_scalar_view<double> inertia,
raft::host_scalar_view<int64_t> n_iter);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const double, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const double, int64_t>>` | Optional weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `std::optional<raft::device_matrix_view<double, int64_t>>` | Optional [in] When init is InitMethod::Array, use centroids  as the initial cluster centers [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<int64_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |
| `inertia` | out | `raft::host_scalar_view<double>` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `raft::host_scalar_view<int64_t>` | Number of iterations run. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit_predict`

Compute balanced k-means clustering and predicts cluster index for each sample in the input.

```cpp
void fit_predict(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const float, int64_t> X,
raft::device_matrix_view<float, int64_t> centroids,
raft::device_vector_view<uint32_t, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | inout | `raft::device_matrix_view<float, int64_t>` | Optional [in] When init is InitMethod::Array, use centroids  as the initial cluster centers [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::fit_predict`

Compute balanced k-means clustering and predicts cluster index for each sample in the input.

```cpp
void fit_predict(const raft::resources& handle,
cuvs::cluster::kmeans::balanced_params const& params,
raft::device_matrix_view<const int8_t, int64_t> X,
raft::device_matrix_view<float, int64_t> centroids,
raft::device_vector_view<uint32_t, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle. |
| `params` | in | [`cuvs::cluster::kmeans::balanced_params const&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-balanced-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const int8_t, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | inout | `raft::device_matrix_view<float, int64_t>` | Optional [in] When init is InitMethod::Array, use centroids  as the initial cluster centers [out] The generated centroids from the kmeans algorithm are stored at the address pointed by 'centroids'. [dim = n_clusters x n_features] |
| `labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | Index of the cluster each sample in X belongs to. [len = n_samples] |

**Returns**

`void`

<a id="cluster-kmeans-transform"></a>
### cluster::kmeans::transform

Transform X to a cluster-distance space.

```cpp
void transform(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const float, int> X,
raft::device_matrix_view<const float, int> centroids,
raft::device_matrix_view<float, int> X_new);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const float, int>` | Training instances to cluster. The data must be in row-major format [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `X_new` | out | `raft::device_matrix_view<float, int>` | X transformed in the new space. [dim = n_samples x n_features] |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::transform`

Transform X to a cluster-distance space.

```cpp
void transform(raft::resources const& handle,
const kmeans::params& params,
raft::device_matrix_view<const double, int> X,
raft::device_matrix_view<const double, int> centroids,
raft::device_matrix_view<double, int> X_new);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft handle. |
| `params` | in | [`const kmeans::params&`](/api-reference/cpp-api-cluster-kmeans#cluster-kmeans-params) | Parameters for KMeans model. |
| `X` | in | `raft::device_matrix_view<const double, int>` | Training instances to cluster. The data must be in row-major format [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const double, int>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `X_new` | out | `raft::device_matrix_view<double, int>` | X transformed in the new space. [dim = n_samples x n_features] |

**Returns**

`void`

<a id="cluster-kmeans-cluster-cost"></a>
### cluster::kmeans::cluster_cost

Compute (optionally weighted) cluster cost

```cpp
void cluster_cost(
const raft::resources& handle,
raft::device_matrix_view<const float, int> X,
raft::device_matrix_view<const float, int> centroids,
raft::device_scalar_view<float> cost,
std::optional<raft::device_vector_view<const float, int>> sample_weight = std::nullopt,
std::optional<raft::device_vector_view<const float, int>> X_norm        = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle |
| `X` | in | `raft::device_matrix_view<const float, int>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `cost` | out | `raft::device_scalar_view<float>` | Resulting cluster cost |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const float, int>>` | Optional per-sample weights. [len = n_samples]<br />Default: `std::nullopt`. |
| `X_norm` | in | `std::optional<raft::device_vector_view<const float, int>>` | Optional precomputed squared L2 row norms of X (\|\|x\|\|^2) [n_samples]. When provided, the internal norm computation is skipped.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::cluster_cost`

Compute cluster cost

```cpp
void cluster_cost(
const raft::resources& handle,
raft::device_matrix_view<const double, int> X,
raft::device_matrix_view<const double, int> centroids,
raft::device_scalar_view<double> cost,
std::optional<raft::device_vector_view<const double, int>> sample_weight = std::nullopt,
std::optional<raft::device_vector_view<const double, int>> X_norm        = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle |
| `X` | in | `raft::device_matrix_view<const double, int>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const double, int>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `cost` | out | `raft::device_scalar_view<double>` | Resulting cluster cost |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const double, int>>` | Optional per-sample weights. [len = n_samples]<br />Default: `std::nullopt`. |
| `X_norm` | in | `std::optional<raft::device_vector_view<const double, int>>` | Optional precomputed squared L2 row norms of X (\|\|x\|\|^2, i.e. sum of squares without the sqrt) [n_samples]. When provided, the internal norm computation is skipped.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::cluster_cost`

Compute (optionally weighted) cluster cost

```cpp
void cluster_cost(
const raft::resources& handle,
raft::device_matrix_view<const float, int64_t> X,
raft::device_matrix_view<const float, int64_t> centroids,
raft::device_scalar_view<float> cost,
std::optional<raft::device_vector_view<const float, int64_t>> sample_weight = std::nullopt,
std::optional<raft::device_vector_view<const float, int64_t>> X_norm        = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `cost` | out | `raft::device_scalar_view<float>` | Resulting cluster cost |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const float, int64_t>>` | Optional per-sample weights. [len = n_samples]<br />Default: `std::nullopt`. |
| `X_norm` | in | `std::optional<raft::device_vector_view<const float, int64_t>>` | Optional precomputed squared L2 row norms of X (\|\|x\|\|^2, i.e. sum of squares without the sqrt) [n_samples]. When provided, the internal norm computation is skipped.<br />Default: `std::nullopt`. |

**Returns**

`void`

**Additional overload:** `cluster::kmeans::cluster_cost`

Compute (optionally weighted) cluster cost

```cpp
void cluster_cost(
const raft::resources& handle,
raft::device_matrix_view<const double, int64_t> X,
raft::device_matrix_view<const double, int64_t> centroids,
raft::device_scalar_view<double> cost,
std::optional<raft::device_vector_view<const double, int64_t>> sample_weight = std::nullopt,
std::optional<raft::device_vector_view<const double, int64_t>> X_norm        = std::nullopt);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | The raft handle |
| `X` | in | `raft::device_matrix_view<const double, int64_t>` | Training instances to cluster. The data must be in row-major format. [dim = n_samples x n_features] |
| `centroids` | in | `raft::device_matrix_view<const double, int64_t>` | Cluster centroids. The data must be in row-major format. [dim = n_clusters x n_features] |
| `cost` | out | `raft::device_scalar_view<double>` | Resulting cluster cost |
| `sample_weight` | in | `std::optional<raft::device_vector_view<const double, int64_t>>` | Optional per-sample weights. [len = n_samples]<br />Default: `std::nullopt`. |
| `X_norm` | in | `std::optional<raft::device_vector_view<const double, int64_t>>` | Optional precomputed squared L2 row norms of X (\|\|x\|\|^2, i.e. sum of squares without the sqrt) [n_samples]. When provided, the internal norm computation is skipped.<br />Default: `std::nullopt`. |

**Returns**

`void`

## k-means API helpers

<a id="cluster-kmeans-helpers-find-k"></a>
### cluster::kmeans::helpers::find_k

Automatically find the optimal value of k using a binary search. This method maximizes the Calinski-Harabasz Index while minimizing the per-cluster inertia.

```cpp
void find_k(raft::resources const& handle,
raft::device_matrix_view<const float, int> X,
raft::host_scalar_view<int> best_k,
raft::host_scalar_view<float> inertia,
raft::host_scalar_view<int> n_iter,
int kmax,
int kmin    = 1,
int maxiter = 100,
float tol   = 1e-3);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` | raft handle |
| `X` |  | `raft::device_matrix_view<const float, int>` | input observations (shape n_samples, n_dims) |
| `best_k` |  | `raft::host_scalar_view<int>` | best k found from binary search |
| `inertia` |  | `raft::host_scalar_view<float>` | inertia of best k found |
| `n_iter` |  | `raft::host_scalar_view<int>` | number of iterations used to find best k |
| `kmax` |  | `int` | maximum k to try in search |
| `kmin` |  | `int` | minimum k to try in search (should be &gt;= 1)<br />Default: `1`. |
| `maxiter` |  | `int` | maximum number of iterations to run<br />Default: `100`. |
| `tol` |  | `float` | tolerance for early stopping convergence<br />Default: `1e-3`. |

**Returns**

`void`
