---
slug: api-reference/python-api-preprocessing-pca
---

# PCA

_Python module: `cuvs.preprocessing.pca`_

## Params

```python
cdef class Params
```

Parameters for PCA decomposition.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `n_components` | `int` | Number of principal components to keep (default: 1). |
| `copy` | `bool` | If False, data passed to fit are overwritten and running fit(X) then transform(X) will not yield the expected results; use fit_transform(X) instead (default: True). |
| `whiten` | `bool` | When True the component vectors are multiplied by the square root of n_samples and divided by the singular values to ensure uncorrelated outputs with unit component-wise variances (default: False). |
| `algorithm` | `str` | Solver algorithm. One of ``"cov_eig_dq"`` (divide-and-conquer) or ``"cov_eig_jacobi"`` (Jacobi) (default: ``"cov_eig_dq"``). |
| `tol` | `float` | Tolerance for singular values, used by the Jacobi solver (default: 0.0). |
| `n_iterations` | `int` | Number of iterations for the Jacobi solver (default: 15). |

**Constructor**

```python
def __init__(self, *, n_components=None, copy=None, whiten=None, algorithm=None, tol=None, n_iterations=None)
```

**Members**

| Name | Kind |
| --- | --- |
| `n_components` | property |
| `copy` | property |
| `whiten` | property |
| `algorithm` | property |
| `tol` | property |
| `n_iterations` | property |

### n_components

```python
def n_components(self)
```

### copy

```python
def copy(self)
```

### whiten

```python
def whiten(self)
```

### algorithm

```python
def algorithm(self)
```

### tol

```python
def tol(self)
```

### n_iterations

```python
def n_iterations(self)
```

## fit

`@auto_sync_resources`

```python
def fit(Params params, X, resources=None)
```

Compute PCA (fit only).

Computes the principal components, explained variances, singular
values, and column means from the input data.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `Params` | PCA parameters. ``params.copy`` should be True if you intend to reuse *X* after this call. |
| `X` | `device array-like, shape (n_samples, n_features), float32` | Input data (will be converted to col-major device memory). |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

FitOutput
Named tuple with fields: ``components``, ``explained_var``,
``explained_var_ratio``, ``singular_vals``, ``mu``,
``noise_vars``.

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing import pca
>>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
>>> params = pca.Params(n_components=8, copy=True)
>>> result = pca.fit(params, X)
>>> result.components.shape
(8, 32)
```

## fit_transform

`@auto_sync_resources`

```python
def fit_transform(Params params, X, resources=None)
```

Compute PCA and transform the input data in a single operation.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `Params` | PCA parameters. |
| `X` | `device array-like, shape (n_samples, n_features), float32` | Input data (will be converted to col-major device memory). |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

FitTransformOutput
Named tuple with fields: ``trans_input``, ``components``,
``explained_var``, ``explained_var_ratio``, ``singular_vals``,
``mu``, ``noise_vars``.

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing import pca
>>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
>>> params = pca.Params(n_components=8)
>>> result = pca.fit_transform(params, X)
>>> result.trans_input.shape
(500, 8)
```

## inverse_transform

`@auto_sync_resources`
`@auto_convert_output`

```python
def inverse_transform(Params params, trans_input, components, singular_vals, mu, output=None, resources=None)
```

Transform data from the PCA eigenspace back to the original space.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `Params` | PCA parameters (must match those used during fit). |
| `trans_input` | `device array-like, shape (n_samples, n_components)` | Transformed data from transform or fit_transform. |
| `components` | `device array-like, shape (n_components, n_features)` | Principal components from a prior fit. |
| `singular_vals` | `device array-like, shape (n_components,)` | Singular values from a prior fit. |
| `mu` | `device array-like, shape (n_features,)` | Column means from a prior fit. |
| `output` | `optional device array, shape (n_samples, n_features)` | Pre-allocated output buffer (col-major, float32). |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `output` | `device array, shape (n_samples, n_features)` | Reconstructed data. |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing import pca
>>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
>>> params = pca.Params(n_components=8)
>>> result = pca.fit_transform(params, X)
>>> reconstructed = pca.inverse_transform(
...     params, result.trans_input, result.components,
...     result.singular_vals, result.mu)
```

## transform

`@auto_sync_resources`
`@auto_convert_output`

```python
def transform(Params params, X, components, singular_vals, mu, trans_input=None, resources=None)
```

Transform data into the PCA eigenspace.

Uses previously computed principal components from fit or
fit_transform.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `params` | `Params` | PCA parameters (must match those used during fit). |
| `X` | `device array-like, shape (n_samples, n_features), float32` | Data to transform. |
| `components` | `device array-like, shape (n_components, n_features)` | Principal components from a prior fit. |
| `singular_vals` | `device array-like, shape (n_components,)` | Singular values from a prior fit. |
| `mu` | `device array-like, shape (n_features,)` | Column means from a prior fit. |
| `trans_input` | `optional device array, shape (n_samples, n_components)` | Pre-allocated output buffer (col-major, float32). |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `trans_input` | `device array, shape (n_samples, n_components)` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.preprocessing import pca
>>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
>>> params = pca.Params(n_components=8, copy=True)
>>> result = pca.fit(params, X)
>>> transformed = pca.transform(params, X, result.components,
...                             result.singular_vals, result.mu)
```
