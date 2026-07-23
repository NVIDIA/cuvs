---
slug: api-reference/cpp-api-cluster-gmm
---

# Gmm

_Source header: `cuvs/cluster/gmm.hpp`_

## Gaussian mixture hyperparameters

<a id="cluster-gmm-covariance-type"></a>
### cluster::gmm::covariance_type

Covariance parameterization of the mixture components.

```cpp
enum class covariance_type {
  FULL = 0,
  TIED = 1,
  DIAG = 2,
  SPHERICAL = 3
};
```

**Values**

| Name | Value |
| --- | --- |
| `FULL` | `0` |
| `TIED` | `1` |
| `DIAG` | `2` |
| `SPHERICAL` | `3` |

<a id="cluster-gmm-init-method"></a>
### cluster::gmm::init_method

Strategy used to initialize the responsibilities before EM.

```cpp
enum class init_method {
  KMeans = 0,
  KMeansPlusPlus = 1,
  Random = 2,
  RandomFromData = 3
};
```

**Values**

| Name | Value |
| --- | --- |
| `KMeans` | `0` |
| `KMeansPlusPlus` | `1` |
| `Random` | `2` |
| `RandomFromData` | `3` |

<a id="cluster-gmm-params"></a>
### cluster::gmm::params

Hyper-parameters for the Gaussian mixture EM solver.

```cpp
struct params {
  int n_components;
  covariance_type cov_type;
  double tol;
  double reg_covar;
  int max_iter;
  int n_init;
  init_method init;
  uint64_t seed;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_components` | `int` | The number of mixture components (at most 65535). Default: 1. |
| `cov_type` | [`covariance_type`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-covariance-type) | Covariance parameterization of the mixture components. Default: FULL. |
| `tol` | `double` | Convergence threshold on the change of the per-sample average log-likelihood (lower bound). Default: 1e-3. |
| `reg_covar` | `double` | Non-negative regularization added to the diagonal of covariance.<br />Default: 1e-6. |
| `max_iter` | `int` | Maximum number of EM iterations for a single run. Default: 100. |
| `n_init` | `int` | Number of initializations to perform; the best result is kept.<br />Default: 1. |
| `init` | [`init_method`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-init-method) | Strategy used to initialize the responsibilities before EM.<br />Default: KMeans. |
| `seed` | `uint64_t` | Seed to the random number generator. Default: 0. |

## Gaussian mixture model APIs

<a id="cluster-gmm-fit"></a>
### cluster::gmm::fit

Fit a Gaussian mixture with the EM algorithm.

```cpp
void fit(raft::resources const& handle,
const params& params,
raft::device_matrix_view<const float, int64_t> X,
raft::device_vector_view<float, int64_t> weights,
raft::device_matrix_view<float, int64_t> means,
raft::device_vector_view<float, int64_t> covariances,
raft::device_vector_view<float, int64_t> precisions_chol,
raft::device_vector_view<float, int64_t> precisions,
raft::device_vector_view<int, int64_t> labels,
raft::host_scalar_view<float> lower_bound,
raft::host_scalar_view<int> n_iter,
raft::host_scalar_view<bool> converged,
bool warm_start = false);
```

Runs ``params.n_init`` random restarts (unless `warm_start` is true) and keeps the parameters with the largest lower bound. Writes the fitted ``weights``, ``means``, ``covariances``, ``precisions_chol`` and ``precisions``, the per-sample hard ``labels`` (argmax of the final responsibilities), and the scalar ``lower_bound`` / ``n_iter`` / ``converged`` diagnostics.

When `warm_start` is true the incoming ``weights`` / ``means`` / ``covariances`` are used as the single initialization and ``params.n_init`` is ignored.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft resources handle. |
| `params` | in | [`const params&`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-params) | Hyper-parameters of the EM solver. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Training data, row-major. [dim = n_samples x n_features] |
| `weights` | inout | `raft::device_vector_view<float, int64_t>` | Mixture weights. [len = n_components] |
| `means` | inout | `raft::device_matrix_view<float, int64_t>` | Component means, row-major. [dim = n_components x n_features] |
| `covariances` | inout | `raft::device_vector_view<float, int64_t>` | Component covariances, flat. Length depends on cov_type (K=n_components, d=n_features): FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K. |
| `precisions_chol` | out | `raft::device_vector_view<float, int64_t>` | Precision Cholesky factors, same flat layout as covariances. FULL/TIED hold the upper-triangular factor U (precision = U @ Uᵀ); DIAG/SPHERICAL hold reciprocal standard deviations. |
| `precisions` | out | `raft::device_vector_view<float, int64_t>` | Precision matrices, same flat layout as covariances. |
| `labels` | out | `raft::device_vector_view<int, int64_t>` | Hard component assignment per sample. [len = n_samples] |
| `lower_bound` | out | `raft::host_scalar_view<float>` | Per-sample average log-likelihood of the best fit. |
| `n_iter` | out | `raft::host_scalar_view<int>` | Number of EM iterations of the best fit. |
| `converged` | out | `raft::host_scalar_view<bool>` | Whether the best fit converged within ``params.tol``. |
| `warm_start` | in | `bool` | Use the incoming weights/means/covariances as the single initialization.<br />Default: `false`. |

**Returns**

`void`

**Additional overload:** `cluster::gmm::fit`

```cpp
void fit(raft::resources const& handle,
const params& params,
raft::device_matrix_view<const double, int64_t> X,
raft::device_vector_view<double, int64_t> weights,
raft::device_matrix_view<double, int64_t> means,
raft::device_vector_view<double, int64_t> covariances,
raft::device_vector_view<double, int64_t> precisions_chol,
raft::device_vector_view<double, int64_t> precisions,
raft::device_vector_view<int, int64_t> labels,
raft::host_scalar_view<double> lower_bound,
raft::host_scalar_view<int> n_iter,
raft::host_scalar_view<bool> converged,
bool warm_start = false);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const params&`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-params) |  |
| `X` |  | `raft::device_matrix_view<const double, int64_t>` |  |
| `weights` |  | `raft::device_vector_view<double, int64_t>` |  |
| `means` |  | `raft::device_matrix_view<double, int64_t>` |  |
| `covariances` |  | `raft::device_vector_view<double, int64_t>` |  |
| `precisions_chol` |  | `raft::device_vector_view<double, int64_t>` |  |
| `precisions` |  | `raft::device_vector_view<double, int64_t>` |  |
| `labels` |  | `raft::device_vector_view<int, int64_t>` |  |
| `lower_bound` |  | `raft::host_scalar_view<double>` |  |
| `n_iter` |  | `raft::host_scalar_view<int>` |  |
| `converged` |  | `raft::host_scalar_view<bool>` |  |
| `warm_start` |  | `bool` | Default: `false`. |

**Returns**

`void`

<a id="cluster-gmm-predict"></a>
### cluster::gmm::predict

Hard component labels (argmax responsibility) for new data.

```cpp
void predict(raft::resources const& handle,
const params& params,
raft::device_matrix_view<const float, int64_t> X,
raft::device_vector_view<const float, int64_t> weights,
raft::device_matrix_view<const float, int64_t> means,
raft::device_vector_view<const float, int64_t> precisions_chol,
raft::device_vector_view<int, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft resources handle. |
| `params` | in | [`const params&`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-params) | Fit hyper-parameters; only n_components and cov_type are consulted at inference time. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Data to assign, row-major. [dim = n_samples x n_features] |
| `weights` | in | `raft::device_vector_view<const float, int64_t>` | Fitted mixture weights. [len = n_components] |
| `means` | in | `raft::device_matrix_view<const float, int64_t>` | Fitted component means. [dim = n_components x n_features] |
| `precisions_chol` | in | `raft::device_vector_view<const float, int64_t>` | Fitted precision Cholesky factors, flat. Length by cov_type (K=n_components, d=n_features): FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K. |
| `labels` | out | `raft::device_vector_view<int, int64_t>` | Hard component assignment per sample. [len = n_samples] |

**Returns**

`void`

**Additional overload:** `cluster::gmm::predict`

```cpp
void predict(raft::resources const& handle,
const params& params,
raft::device_matrix_view<const double, int64_t> X,
raft::device_vector_view<const double, int64_t> weights,
raft::device_matrix_view<const double, int64_t> means,
raft::device_vector_view<const double, int64_t> precisions_chol,
raft::device_vector_view<int, int64_t> labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const params&`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-params) |  |
| `X` |  | `raft::device_matrix_view<const double, int64_t>` |  |
| `weights` |  | `raft::device_vector_view<const double, int64_t>` |  |
| `means` |  | `raft::device_matrix_view<const double, int64_t>` |  |
| `precisions_chol` |  | `raft::device_vector_view<const double, int64_t>` |  |
| `labels` |  | `raft::device_vector_view<int, int64_t>` |  |

**Returns**

`void`

<a id="cluster-gmm-predict-proba"></a>
### cluster::gmm::predict_proba

Posterior responsibilities for new data.

```cpp
void predict_proba(raft::resources const& handle,
const params& params,
raft::device_matrix_view<const float, int64_t> X,
raft::device_vector_view<const float, int64_t> weights,
raft::device_matrix_view<const float, int64_t> means,
raft::device_vector_view<const float, int64_t> precisions_chol,
raft::device_matrix_view<float, int64_t> resp);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft resources handle. |
| `params` | in | [`const params&`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-params) | Fit hyper-parameters; only n_components and cov_type are consulted at inference time. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Data to evaluate, row-major. [dim = n_samples x n_features] |
| `weights` | in | `raft::device_vector_view<const float, int64_t>` | Fitted mixture weights. [len = n_components] |
| `means` | in | `raft::device_matrix_view<const float, int64_t>` | Fitted component means. [dim = n_components x n_features] |
| `precisions_chol` | in | `raft::device_vector_view<const float, int64_t>` | Fitted precision Cholesky factors, flat. Length by cov_type (K=n_components, d=n_features): FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K. |
| `resp` | out | `raft::device_matrix_view<float, int64_t>` | Posterior probability of each component for each sample, row-major. [dim = n_samples x n_components] |

**Returns**

`void`

**Additional overload:** `cluster::gmm::predict_proba`

```cpp
void predict_proba(raft::resources const& handle,
const params& params,
raft::device_matrix_view<const double, int64_t> X,
raft::device_vector_view<const double, int64_t> weights,
raft::device_matrix_view<const double, int64_t> means,
raft::device_vector_view<const double, int64_t> precisions_chol,
raft::device_matrix_view<double, int64_t> resp);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const params&`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-params) |  |
| `X` |  | `raft::device_matrix_view<const double, int64_t>` |  |
| `weights` |  | `raft::device_vector_view<const double, int64_t>` |  |
| `means` |  | `raft::device_matrix_view<const double, int64_t>` |  |
| `precisions_chol` |  | `raft::device_vector_view<const double, int64_t>` |  |
| `resp` |  | `raft::device_matrix_view<double, int64_t>` |  |

**Returns**

`void`

<a id="cluster-gmm-score-samples"></a>
### cluster::gmm::score_samples

Per-sample log-likelihood log p(x_i) for new data.

```cpp
void score_samples(raft::resources const& handle,
const params& params,
raft::device_matrix_view<const float, int64_t> X,
raft::device_vector_view<const float, int64_t> weights,
raft::device_matrix_view<const float, int64_t> means,
raft::device_vector_view<const float, int64_t> precisions_chol,
raft::device_vector_view<float, int64_t> log_prob_norm);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | The raft resources handle. |
| `params` | in | [`const params&`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-params) | Fit hyper-parameters; only n_components and cov_type are consulted at inference time. |
| `X` | in | `raft::device_matrix_view<const float, int64_t>` | Data to evaluate, row-major. [dim = n_samples x n_features] |
| `weights` | in | `raft::device_vector_view<const float, int64_t>` | Fitted mixture weights. [len = n_components] |
| `means` | in | `raft::device_matrix_view<const float, int64_t>` | Fitted component means. [dim = n_components x n_features] |
| `precisions_chol` | in | `raft::device_vector_view<const float, int64_t>` | Fitted precision Cholesky factors, flat. Length by cov_type (K=n_components, d=n_features): FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K. |
| `log_prob_norm` | out | `raft::device_vector_view<float, int64_t>` | Log-likelihood of each sample under the model. [len = n_samples] |

**Returns**

`void`

**Additional overload:** `cluster::gmm::score_samples`

```cpp
void score_samples(raft::resources const& handle,
const params& params,
raft::device_matrix_view<const double, int64_t> X,
raft::device_vector_view<const double, int64_t> weights,
raft::device_matrix_view<const double, int64_t> means,
raft::device_vector_view<const double, int64_t> precisions_chol,
raft::device_vector_view<double, int64_t> log_prob_norm);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `params` |  | [`const params&`](/api-reference/cpp-api-cluster-gmm#cluster-gmm-params) |  |
| `X` |  | `raft::device_matrix_view<const double, int64_t>` |  |
| `weights` |  | `raft::device_vector_view<const double, int64_t>` |  |
| `means` |  | `raft::device_matrix_view<const double, int64_t>` |  |
| `precisions_chol` |  | `raft::device_vector_view<const double, int64_t>` |  |
| `log_prob_norm` |  | `raft::device_vector_view<double, int64_t>` |  |

**Returns**

`void`
