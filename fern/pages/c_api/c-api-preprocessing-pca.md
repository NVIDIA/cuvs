---
slug: api-reference/c-api-preprocessing-pca
---

# PCA

_Source header: `cuvs/preprocessing/pca.h`_

## C API for PCA (Principal Component Analysis)

<a id="cuvspcasolver"></a>
### cuvsPcaSolver

Solver algorithm for PCA eigen decomposition.

```c
enum cuvsPcaSolver {
  CUVS_PCA_COV_EIG_DQ = 0,
  CUVS_PCA_COV_EIG_JACOBI = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_PCA_COV_EIG_DQ` | `0` |
| `CUVS_PCA_COV_EIG_JACOBI` | `1` |

<a id="cuvspcaparams"></a>
### cuvsPcaParams

Parameters for PCA decomposition.

```c
struct cuvsPcaParams {
  int n_components;
  bool copy;
  bool whiten;
  enum cuvsPcaSolver algorithm;
  float tol;
  int n_iterations;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_components` | `int` | Number of principal components to keep. |
| `copy` | `bool` | If false, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results; use fit_transform(X) instead. |
| `whiten` | `bool` | When true the component vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. |
| `algorithm` | [`enum cuvsPcaSolver`](/api-reference/c-api-preprocessing-pca#cuvspcasolver) | Solver algorithm to use. |
| `tol` | `float` | Tolerance for singular values (used by Jacobi solver). |
| `n_iterations` | `int` | Number of iterations for the power method (Jacobi solver). |

<a id="cuvspcaparamscreate"></a>
### cuvsPcaParamsCreate

Allocate PCA params and populate with default values.

```c
cuvsError_t cuvsPcaParamsCreate(cuvsPcaParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | out | [`cuvsPcaParams_t*`](/api-reference/c-api-preprocessing-pca#cuvspcaparams) | cuvsPcaParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvspcaparamsdestroy"></a>
### cuvsPcaParamsDestroy

De-allocate PCA params.

```c
cuvsError_t cuvsPcaParamsDestroy(cuvsPcaParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsPcaParams_t`](/api-reference/c-api-preprocessing-pca#cuvspcaparams) | cuvsPcaParams_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvspcafit"></a>
### cuvsPcaFit

Perform PCA fit operation.

```c
cuvsError_t cuvsPcaFit(cuvsResources_t res,
cuvsPcaParams_t params,
DLManagedTensor* input,
DLManagedTensor* components,
DLManagedTensor* explained_var,
DLManagedTensor* explained_var_ratio,
DLManagedTensor* singular_vals,
DLManagedTensor* mu,
DLManagedTensor* noise_vars,
bool flip_signs_based_on_U);
```

Computes the principal components, explained variances, singular values, and column means from the input data.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsPcaParams_t`](/api-reference/c-api-preprocessing-pca#cuvspcaparams) | PCA parameters |
| `input` | inout | `DLManagedTensor*` | input data [n_rows x n_cols] (col-major, float32, device) |
| `components` | out | `DLManagedTensor*` | principal components [n_components x n_cols] (col-major, float32, device) |
| `explained_var` | out | `DLManagedTensor*` | explained variances [n_components] (float32, device) |
| `explained_var_ratio` | out | `DLManagedTensor*` | explained variance ratios [n_components] (float32, device) |
| `singular_vals` | out | `DLManagedTensor*` | singular values [n_components] (float32, device) |
| `mu` | out | `DLManagedTensor*` | column means [n_cols] (float32, device) |
| `noise_vars` | out | `DLManagedTensor*` | noise variance [1] (float32, device) |
| `flip_signs_based_on_U` | in | `bool` | whether to determine signs by U (true) or V.T (false) |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvspcafittransform"></a>
### cuvsPcaFitTransform

Perform PCA fit and transform in a single operation.

```c
cuvsError_t cuvsPcaFitTransform(cuvsResources_t res,
cuvsPcaParams_t params,
DLManagedTensor* input,
DLManagedTensor* trans_input,
DLManagedTensor* components,
DLManagedTensor* explained_var,
DLManagedTensor* explained_var_ratio,
DLManagedTensor* singular_vals,
DLManagedTensor* mu,
DLManagedTensor* noise_vars,
bool flip_signs_based_on_U);
```

Computes the principal components and transforms the input data into the eigenspace.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsPcaParams_t`](/api-reference/c-api-preprocessing-pca#cuvspcaparams) | PCA parameters |
| `input` | inout | `DLManagedTensor*` | input data [n_rows x n_cols] (col-major, float32, device) |
| `trans_input` | out | `DLManagedTensor*` | transformed data [n_rows x n_components] (col-major, float32, device) |
| `components` | out | `DLManagedTensor*` | principal components [n_components x n_cols] (col-major, float32, device) |
| `explained_var` | out | `DLManagedTensor*` | explained variances [n_components] (float32, device) |
| `explained_var_ratio` | out | `DLManagedTensor*` | explained variance ratios [n_components] (float32, device) |
| `singular_vals` | out | `DLManagedTensor*` | singular values [n_components] (float32, device) |
| `mu` | out | `DLManagedTensor*` | column means [n_cols] (float32, device) |
| `noise_vars` | out | `DLManagedTensor*` | noise variance [1] (float32, device) |
| `flip_signs_based_on_U` | in | `bool` | whether to determine signs by U (true) or V.T (false) |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvspcatransform"></a>
### cuvsPcaTransform

Perform PCA transform operation.

```c
cuvsError_t cuvsPcaTransform(cuvsResources_t res,
cuvsPcaParams_t params,
DLManagedTensor* input,
DLManagedTensor* components,
DLManagedTensor* singular_vals,
DLManagedTensor* mu,
DLManagedTensor* trans_input);
```

Transforms the input data into the eigenspace using previously computed principal components.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsPcaParams_t`](/api-reference/c-api-preprocessing-pca#cuvspcaparams) | PCA parameters |
| `input` | inout | `DLManagedTensor*` | data to transform [n_rows x n_cols] (col-major, float32, device) |
| `components` | in | `DLManagedTensor*` | principal components [n_components x n_cols] (col-major, float32, device) |
| `singular_vals` | in | `DLManagedTensor*` | singular values [n_components] (float32, device) |
| `mu` | in | `DLManagedTensor*` | column means [n_cols] (float32, device) |
| `trans_input` | out | `DLManagedTensor*` | transformed data [n_rows x n_components] (col-major, float32, device) |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvspcainversetransform"></a>
### cuvsPcaInverseTransform

Perform PCA inverse transform operation.

```c
cuvsError_t cuvsPcaInverseTransform(cuvsResources_t res,
cuvsPcaParams_t params,
DLManagedTensor* trans_input,
DLManagedTensor* components,
DLManagedTensor* singular_vals,
DLManagedTensor* mu,
DLManagedTensor* output);
```

Transforms data from the eigenspace back to the original space.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsPcaParams_t`](/api-reference/c-api-preprocessing-pca#cuvspcaparams) | PCA parameters |
| `trans_input` | in | `DLManagedTensor*` | transformed data [n_rows x n_components] (col-major, float32, device) |
| `components` | in | `DLManagedTensor*` | principal components [n_components x n_cols] (col-major, float32, device) |
| `singular_vals` | in | `DLManagedTensor*` | singular values [n_components] (float32, device) |
| `mu` | in | `DLManagedTensor*` | column means [n_cols] (float32, device) |
| `output` | out | `DLManagedTensor*` | reconstructed data [n_rows x n_cols] (col-major, float32, device) |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
