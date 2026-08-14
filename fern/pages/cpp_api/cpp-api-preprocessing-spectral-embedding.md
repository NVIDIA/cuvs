---
slug: api-reference/cpp-api-preprocessing-spectral-embedding
---

# Spectral Embedding

_Source header: `cuvs/preprocessing/spectral_embedding.hpp`_

## Types

<a id="preprocessing-spectral-embedding-params"></a>
### preprocessing::spectral_embedding::params

Parameters for spectral embedding algorithm

Spectral embedding is a dimensionality reduction technique that uses the eigenvectors of the graph Laplacian to embed data points into a lower-dimensional space. This technique is particularly useful for non-linear dimensionality reduction and clustering tasks.

```cpp
struct params {
  int n_components;
  int n_neighbors;
  bool norm_laplacian;
  bool drop_first;
  float tolerance;
  std::optional<uint64_t> seed;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_components` | `int` | The number of components to reduce the data to. |
| `n_neighbors` | `int` | The number of neighbors to use for the nearest neighbors graph. |
| `norm_laplacian` | `bool` | Whether to normalize the Laplacian matrix. If true, uses the normalized graph Laplacian (D^(-1/2) L D^(-1/2)). If false, uses the unnormalized graph Laplacian (L = D - W). Normalized Laplacian often leads to better results for clustering tasks. |
| `drop_first` | `bool` | Whether to drop the first eigenvector. The first eigenvector of the normalized Laplacian is constant and uninformative. Setting this to true drops it from the embedding. This is typically set to true when norm_laplacian is true. |
| `tolerance` | `float` | Tolerance for the eigenvalue solver. The tolerance for the eigenvalue solver. This is used to determine when to stop the eigenvalue solver. |
| `seed` | `std::optional<uint64_t>` | Random seed for reproducibility. Controls the random number generation for k-NN graph construction and eigenvalue solver initialization. Use the same seed value to ensure reproducible results across runs. |

## Spectral Embedding

<a id="preprocessing-spectral-embedding-transform"></a>
### preprocessing::spectral_embedding::transform

Perform spectral embedding on input dataset

```cpp
void transform(raft::resources const& handle,
params config,
raft::device_matrix_view<float, int, raft::row_major> dataset,
raft::device_matrix_view<float, int, raft::col_major> embedding);
```

This function computes the spectral embedding of the input dataset by:

1. Constructing a k-nearest neighbors graph from the input data
2. Computing the graph Laplacian (normalized or unnormalized)
3. Finding the eigenvectors corresponding to the smallest eigenvalues
4. Using these eigenvectors as the embedding coordinates

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | RAFT resource handle for managing CUDA resources |
| `config` | in | [`params`](/api-reference/cpp-api-preprocessing-spectral-embedding#preprocessing-spectral-embedding-params) | Parameters controlling the spectral embedding algorithm |
| `dataset` | in | `raft::device_matrix_view<float, int, raft::row_major>` | Input dataset in row-major format [n_samples x n_features] |
| `embedding` | out | `raft::device_matrix_view<float, int, raft::col_major>` | Output embedding in column-major format [n_samples x n_components] |

**Returns**

`void`

**Additional overload:** `preprocessing::spectral_embedding::transform`

Perform spectral embedding using a precomputed connectivity graph

```cpp
void transform(raft::resources const& handle,
params config,
raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
raft::device_matrix_view<float, int, raft::col_major> embedding);
```

This function computes the spectral embedding from a precomputed sparse connectivity graph (e.g., from a k-NN search or custom similarity matrix). This is useful when you want to use a custom graph construction method or when you have a precomputed similarity/affinity matrix.

The function:

1. Converts the COO matrix to the graph Laplacian
2. Computes eigenvectors of the Laplacian
3. Returns the eigenvectors as the embedding

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | RAFT resource handle for managing CUDA resources |
| `config` | in | [`params`](/api-reference/cpp-api-preprocessing-spectral-embedding#preprocessing-spectral-embedding-params) | Parameters controlling the spectral embedding algorithm (n_neighbors parameter is ignored when using precomputed graph) |
| `connectivity_graph` | in | `raft::device_coo_matrix_view<float, int, int, int>` | Precomputed sparse connectivity/affinity graph in COO format representing weighted connections between samples |
| `embedding` | out | `raft::device_matrix_view<float, int, raft::col_major>` | Output embedding in column-major format [n_samples x n_components] |

**Returns**

`void`
