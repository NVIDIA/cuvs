---
slug: api-reference/cpp-api-preprocessing-pca
---

# PCA

_Source header: `cuvs/preprocessing/pca.hpp`_

## Types

<a id="preprocessing-pca-params"></a>
### preprocessing::pca::params

Parameters for PCA decomposition. Ref: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

```cpp
struct params {
  int n_components;
  bool copy;
  bool whiten;
  solver algorithm;
  float tol;
  int n_iterations;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_components` | `int` | Number of components to keep. |
| `copy` | `bool` | If false, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead. |
| `whiten` | `bool` | When true (false by default) the components vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. |
| `algorithm` | `solver` | The solver algorithm to use. |
| `tol` | `float` | Tolerance for singular values computed by svd_solver == 'arpack' or the Jacobi solver. |
| `n_iterations` | `int` | Number of iterations for the power method computed by the Jacobi solver. |

## PCA (Principal Component Analysis)

<a id="preprocessing-pca-fit"></a>
### preprocessing::pca::fit

Perform PCA fit operation.

```cpp
void fit(raft::resources const& handle,
const params& config,
raft::device_matrix_view<float, int64_t, raft::col_major> input,
raft::device_matrix_view<float, int64_t, raft::col_major> components,
raft::device_vector_view<float, int64_t> explained_var,
raft::device_vector_view<float, int64_t> explained_var_ratio,
raft::device_vector_view<float, int64_t> singular_vals,
raft::device_vector_view<float, int64_t> mu,
raft::device_scalar_view<float, int64_t> noise_vars,
bool flip_signs_based_on_U = false);
```

Computes the principal components, explained variances, singular values, and column means from the input data.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resource handle |
| `config` | in | [`const params&`](/api-reference/cpp-api-preprocessing-pca#preprocessing-pca-params) | PCA parameters |
| `input` | inout | `raft::device_matrix_view<float, int64_t, raft::col_major>` | input data [n_rows x n_cols] (col-major). Modified temporarily. |
| `components` | out | `raft::device_matrix_view<float, int64_t, raft::col_major>` | principal components [n_components x n_cols] (col-major) |
| `explained_var` | out | `raft::device_vector_view<float, int64_t>` | explained variances [n_components] |
| `explained_var_ratio` | out | `raft::device_vector_view<float, int64_t>` | explained variance ratios [n_components] |
| `singular_vals` | out | `raft::device_vector_view<float, int64_t>` | singular values [n_components] |
| `mu` | out | `raft::device_vector_view<float, int64_t>` | column means [n_cols] |
| `noise_vars` | out | `raft::device_scalar_view<float, int64_t>` | noise variance (scalar) |
| `flip_signs_based_on_U` | in | `bool` | whether to determine signs by U (true) or V.T (false)<br />Default: `false`. |

**Returns**

`void`

<a id="preprocessing-pca-fit-transform"></a>
### preprocessing::pca::fit_transform

Perform PCA fit and transform operations.

```cpp
void fit_transform(raft::resources const& handle,
const params& config,
raft::device_matrix_view<float, int64_t, raft::col_major> input,
raft::device_matrix_view<float, int64_t, raft::col_major> trans_input,
raft::device_matrix_view<float, int64_t, raft::col_major> components,
raft::device_vector_view<float, int64_t> explained_var,
raft::device_vector_view<float, int64_t> explained_var_ratio,
raft::device_vector_view<float, int64_t> singular_vals,
raft::device_vector_view<float, int64_t> mu,
raft::device_scalar_view<float, int64_t> noise_vars,
bool flip_signs_based_on_U = false);
```

Computes the principal components and transforms the input data into the eigenspace in a single operation.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resource handle |
| `config` | in | [`const params&`](/api-reference/cpp-api-preprocessing-pca#preprocessing-pca-params) | PCA parameters |
| `input` | inout | `raft::device_matrix_view<float, int64_t, raft::col_major>` | input data [n_rows x n_cols] (col-major). Modified temporarily. |
| `trans_input` | out | `raft::device_matrix_view<float, int64_t, raft::col_major>` | transformed data [n_rows x n_components] (col-major) |
| `components` | out | `raft::device_matrix_view<float, int64_t, raft::col_major>` | principal components [n_components x n_cols] (col-major) |
| `explained_var` | out | `raft::device_vector_view<float, int64_t>` | explained variances [n_components] |
| `explained_var_ratio` | out | `raft::device_vector_view<float, int64_t>` | explained variance ratios [n_components] |
| `singular_vals` | out | `raft::device_vector_view<float, int64_t>` | singular values [n_components] |
| `mu` | out | `raft::device_vector_view<float, int64_t>` | column means [n_cols] |
| `noise_vars` | out | `raft::device_scalar_view<float, int64_t>` | noise variance (scalar) |
| `flip_signs_based_on_U` | in | `bool` | whether to determine signs by U (true) or V.T (false)<br />Default: `false`. |

**Returns**

`void`

<a id="preprocessing-pca-transform"></a>
### preprocessing::pca::transform

Perform PCA transform operation.

```cpp
void transform(raft::resources const& handle,
const params& config,
raft::device_matrix_view<float, int64_t, raft::col_major> input,
raft::device_matrix_view<float, int64_t, raft::col_major> components,
raft::device_vector_view<float, int64_t> singular_vals,
raft::device_vector_view<float, int64_t> mu,
raft::device_matrix_view<float, int64_t, raft::col_major> trans_input);
```

Transforms the input data into the eigenspace using previously computed principal components.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resource handle |
| `config` | in | [`const params&`](/api-reference/cpp-api-preprocessing-pca#preprocessing-pca-params) | PCA parameters |
| `input` | inout | `raft::device_matrix_view<float, int64_t, raft::col_major>` | data to transform [n_rows x n_cols] (col-major). Modified temporarily (mean-centered then restored). |
| `components` | in | `raft::device_matrix_view<float, int64_t, raft::col_major>` | principal components [n_components x n_cols] (col-major) |
| `singular_vals` | in | `raft::device_vector_view<float, int64_t>` | singular values [n_components] |
| `mu` | in | `raft::device_vector_view<float, int64_t>` | column means [n_cols] |
| `trans_input` | out | `raft::device_matrix_view<float, int64_t, raft::col_major>` | transformed data [n_rows x n_components] (col-major) |

**Returns**

`void`

<a id="preprocessing-pca-inverse-transform"></a>
### preprocessing::pca::inverse_transform

Perform PCA inverse transform operation.

```cpp
void inverse_transform(raft::resources const& handle,
const params& config,
raft::device_matrix_view<float, int64_t, raft::col_major> trans_input,
raft::device_matrix_view<float, int64_t, raft::col_major> components,
raft::device_vector_view<float, int64_t> singular_vals,
raft::device_vector_view<float, int64_t> mu,
raft::device_matrix_view<float, int64_t, raft::col_major> output);
```

Transforms data from the eigenspace back to the original space.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resource handle |
| `config` | in | [`const params&`](/api-reference/cpp-api-preprocessing-pca#preprocessing-pca-params) | PCA parameters |
| `trans_input` | in | `raft::device_matrix_view<float, int64_t, raft::col_major>` | transformed data [n_rows x n_components] (col-major) |
| `components` | in | `raft::device_matrix_view<float, int64_t, raft::col_major>` | principal components [n_components x n_cols] (col-major) |
| `singular_vals` | in | `raft::device_vector_view<float, int64_t>` | singular values [n_components] |
| `mu` | in | `raft::device_vector_view<float, int64_t>` | column means [n_cols] |
| `output` | out | `raft::device_matrix_view<float, int64_t, raft::col_major>` | reconstructed data [n_rows x n_cols] (col-major) |

**Returns**

`void`
