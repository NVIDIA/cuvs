# PCA

Principal Component Analysis, or PCA, is a GPU-accelerated dimensionality reduction algorithm. It learns directions of high variance in a dataset and projects each row onto a smaller number of components.

Use PCA when you want to reduce vector dimensionality, denoise data, visualize high-dimensional data, or prepare a lower-dimensional representation before another algorithm. PCA is lossy when `n_components` is smaller than the original feature count.

## Example API Usage

[C API](/api-reference/c-api-preprocessing-pca) | [C++ API](/api-reference/cpp-api-preprocessing-pca) | [Python API](/api-reference/python-api-preprocessing-pca)

### Fitting components

Fitting learns the principal components, explained variances, singular values, and column means from a col-major `float32` input matrix.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/pca.h>

cuvsResources_t res;
cuvsPcaParams_t params;
DLManagedTensor *input;
DLManagedTensor *components;
DLManagedTensor *explained_var;
DLManagedTensor *explained_var_ratio;
DLManagedTensor *singular_vals;
DLManagedTensor *mu;
DLManagedTensor *noise_vars;

load_col_major_dataset(input);
allocate_pca_outputs(components,
                     explained_var,
                     explained_var_ratio,
                     singular_vals,
                     mu,
                     noise_vars);

cuvsResourcesCreate(&res);
cuvsPcaParamsCreate(&params);

params->n_components = 32;
params->copy = true;
params->whiten = false;

cuvsPcaFit(res,
           params,
           input,
           components,
           explained_var,
           explained_var_ratio,
           singular_vals,
           mu,
           noise_vars,
           false);

cuvsPcaParamsDestroy(params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/preprocessing/pca.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace pca = cuvs::preprocessing::pca;

raft::device_resources res;
raft::device_matrix_view<float, int64_t, raft::col_major> input =
    load_col_major_dataset();

pca::params params;
params.n_components = 32;
params.copy = true;
params.whiten = false;

auto components = raft::make_device_matrix<float, int64_t, raft::col_major>(
    res, params.n_components, input.extent(1));
auto explained_var = raft::make_device_vector<float, int64_t>(
    res, params.n_components);
auto explained_var_ratio = raft::make_device_vector<float, int64_t>(
    res, params.n_components);
auto singular_vals = raft::make_device_vector<float, int64_t>(
    res, params.n_components);
auto mu = raft::make_device_vector<float, int64_t>(res, input.extent(1));
auto noise_vars = raft::make_device_scalar<float>(res);

pca::fit(res,
         params,
         input,
         components.view(),
         explained_var.view(),
         explained_var_ratio.view(),
         singular_vals.view(),
         mu.view(),
         noise_vars.view());
```

</Tab>
<Tab title="Python">

```python
import cupy as cp

from cuvs.preprocessing import pca

dataset = cp.asarray(load_dataset(), dtype=cp.float32)
params = pca.Params(n_components=32, copy=True, whiten=False)

fit_result = pca.fit(params, dataset)
```

</Tab>
</Tabs>

### Transforming data

Transforming projects rows into the PCA component space. `fit_transform` combines fitting and transforming in one call.

<Tabs>
<Tab title="C">

```c
DLManagedTensor *transformed;

allocate_col_major_matrix(transformed, n_rows, params->n_components);

cuvsPcaFitTransform(res,
                    params,
                    input,
                    transformed,
                    components,
                    explained_var,
                    explained_var_ratio,
                    singular_vals,
                    mu,
                    noise_vars,
                    false);
```

</Tab>
<Tab title="C++">

```cpp
auto transformed = raft::make_device_matrix<float, int64_t, raft::col_major>(
    res, input.extent(0), params.n_components);

pca::fit_transform(res,
                   params,
                   input,
                   transformed.view(),
                   components.view(),
                   explained_var.view(),
                   explained_var_ratio.view(),
                   singular_vals.view(),
                   mu.view(),
                   noise_vars.view());
```

</Tab>
<Tab title="Python">

```python
result = pca.fit_transform(params, dataset)
transformed = result.trans_input

projected = pca.transform(
    params,
    dataset,
    result.components,
    result.singular_vals,
    result.mu,
)
```

</Tab>
</Tabs>

### Reconstructing data

Inverse transform maps PCA-space rows back to the original feature space. The reconstruction is approximate when fewer components are kept.

<Tabs>
<Tab title="C">

```c
DLManagedTensor *reconstructed;

allocate_col_major_matrix(reconstructed, n_rows, n_features);

cuvsPcaInverseTransform(res,
                        params,
                        transformed,
                        components,
                        singular_vals,
                        mu,
                        reconstructed);
```

</Tab>
<Tab title="C++">

```cpp
auto reconstructed = raft::make_device_matrix<float, int64_t, raft::col_major>(
    res, input.extent(0), input.extent(1));

pca::inverse_transform(res,
                       params,
                       transformed.view(),
                       components.view(),
                       singular_vals.view(),
                       mu.view(),
                       reconstructed.view());
```

</Tab>
<Tab title="Python">

```python
reconstructed = pca.inverse_transform(
    params,
    transformed,
    result.components,
    result.singular_vals,
    result.mu,
)
```

</Tab>
</Tabs>

## How PCA works

PCA centers the input columns and finds orthogonal directions that explain the most variance. Keeping the first `n_components` directions gives a lower-dimensional representation that preserves as much variance as possible under a linear projection.

When `whiten=True`, the transformed components are scaled to have unit component-wise variance. Whitening can help downstream models that assume similarly scaled features, but it also removes the original variance scale.

## When to use PCA

Use PCA when the data has redundant or noisy dimensions and a linear lower-dimensional representation is acceptable. It can reduce memory use, reduce distance-computation cost, and make later algorithms easier to tune.

Avoid PCA when interpretability of original dimensions is required, when nonlinear structure is the main signal, or when dropping low-variance directions would remove important information.

## Configuration parameters

### Fit parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `n_components` | `1` | Number of principal components to keep. Larger values preserve more information but reduce memory and compute less. |
| `copy` | `true` | Preserves the input during fit. If false, fit may temporarily overwrite input data. |
| `whiten` | `false` | Scales transformed components to unit variance. |
| `algorithm` | `cov_eig_dq` | PCA solver. Python accepts `cov_eig_dq` or `cov_eig_jacobi`; C uses `CUVS_PCA_COV_EIG_DQ` or `CUVS_PCA_COV_EIG_JACOBI`. |
| `tol` | `0.0` | Tolerance used by the Jacobi solver. |
| `n_iterations` | `15` | Number of Jacobi solver iterations. |
| `flip_signs_based_on_U` | `false` | Controls the sign convention for fitted components in C and C++. |

## Tuning

Start with `n_components` based on the target dimensionality or the amount of variance you need to preserve. Increase it when reconstruction quality or downstream accuracy is too low.

Use the default divide-and-conquer covariance eigensolver for most workloads. Try the Jacobi solver when you need its convergence behavior, then tune `tol` and `n_iterations` together.

Enable `whiten` only when the downstream workflow benefits from unit-variance components. Whitening changes component scaling, so compare downstream metrics before making it the default.

## Memory footprint

PCA memory is dominated by the input matrix, the covariance workspace, the component matrix, and the transformed matrix when `fit_transform` is used.

Variables:

- `N`: Number of rows.
- `D`: Number of input features.
- `K`: Number of retained components.
- `B_x`: Bytes per floating-point element.

### Scratch and maximum rows

The `scratch` term covers covariance or solver workspace, temporary centered data, allocator padding, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.30` for PCA fit and `fit_transform`, because eigensolver workspace can be significant. If you can measure a representative run, use:

$$
H_{\text{measured}}
  =
  \frac{\text{observed\_peak} - \text{formula\_without\_scratch}}
       {\text{formula\_without\_scratch}}
$$

Then set:

$$
M_{\text{usable}}
  = (M_{\text{free}} - M_{\text{other}}) \cdot (1 - H)
$$

The capacity variables in this subsection are:

- `M_free`: Free memory in the relevant memory space before the operation starts. Use device memory for GPU-resident formulas and host memory for formulas explicitly marked as host memory.
- `M_other`: Memory reserved for arrays, memory pools, concurrent work, or application buffers that are not included in the formula.
- `H`: Scratch headroom fraction reserved for temporary buffers and allocator overhead.
- `M_usable`: Memory budget left for the formula after subtracting `M_other` and reserving headroom.
- `observed_peak`: Peak memory observed during a smaller representative run.
- `formula_without_scratch`: Value of the selected peak formula with explicit `scratch` terms removed and without applying headroom.
- `peak_without_scratch(count)`: The selected peak formula rewritten as a function of the count being estimated, excluding scratch and headroom. The count is usually `N` for rows or vectors and `B` for K-selection batch rows.
- `B_per_row` / `B_per_vector`: Bytes added by one more row or vector in the selected formula. For linear formulas, add the coefficients of the count being estimated after fixed values such as `D`, `K`, `Q`, and `L` are substituted.
- `B_fixed`: Bytes in the selected formula that do not change with the estimated count, such as codebooks, centroids, fixed query batches, capped training buffers, or metadata.
- `N_max` / `B_max`: Estimated largest row, vector, or batch-row count that fits in `M_usable`.


For fixed `D` and `K`, solve the fit peak as a linear function of `N`:

$$
\text{peak\_without\_scratch}(N)
  = N \cdot B_{\text{per\_row}} + B_{\text{fixed}}
$$

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_row}}}
  \right\rfloor
$$

The covariance workspace scales with `D^2`, so it belongs in `B_fixed` when `D` is fixed. If `D` also changes, solve the full formula rather than using the linear shortcut.

### Persistent arrays

The main arrays are:

$$
\begin{aligned}
\text{input\_size} &= N \cdot D \cdot B_x \\
\text{components\_size} &= K \cdot D \cdot B_x \\
\text{transformed\_size} &= N \cdot K \cdot B_x
\end{aligned}
$$

The vectors for explained variance, explained variance ratio, singular values, means, and noise variance are smaller:

$$
\text{stats\_size}
  \approx (3K + D + 1) \cdot B_x
$$

### Fit peak

The covariance-based solvers can require a feature-by-feature covariance workspace:

$$
\text{covariance\_size}
  \approx D^2 \cdot B_x
$$

The fit-transform peak is approximately:

$$
\begin{aligned}
\text{fit\_transform\_peak}
  \approx&\ \text{input\_size}
   + \text{covariance\_size} \\
  &+ \text{components\_size}
   + \text{transformed\_size}
   + \text{stats\_size}
   + \text{scratch}
\end{aligned}
$$

For very high-dimensional data, reduce `n_components` only reduces the component and transformed matrices. The covariance workspace still scales with `D^2`.
