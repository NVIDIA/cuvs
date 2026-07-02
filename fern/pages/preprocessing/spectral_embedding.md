# Spectral Embedding

Spectral Embedding is a GPU-accelerated preprocessing algorithm for graph-based dimensionality reduction. It builds or consumes a sparse connectivity graph, computes a graph Laplacian, and uses eigenvectors of that Laplacian as a lower-dimensional representation of the data.

Use Spectral Embedding when neighborhood structure matters more than a purely linear projection. It is useful for exploratory analysis, visualization, manifold-style workflows, and clustering pipelines where nearby points should remain close in the embedding.

## Example API Usage

[C++ API](/api-reference/cpp-api-preprocessing-spectral-embedding)

### Computing an embedding

The dense-data path builds a nearest-neighbor connectivity graph from the input matrix, then computes the spectral embedding.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/preprocessing/spectral_embedding.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace spectral_embedding = cuvs::preprocessing::spectral_embedding;

raft::device_resources res;
raft::device_matrix_view<float, int, raft::row_major> dataset = load_dataset();

spectral_embedding::params params;
params.n_components = 2;
params.n_neighbors = 15;
params.norm_laplacian = true;
params.drop_first = true;
params.tolerance = 1e-5f;
params.seed = 42;

auto embedding = raft::make_device_matrix<float, int, raft::col_major>(
    res, dataset.extent(0), params.n_components);

spectral_embedding::transform(res, params, dataset, embedding.view());
```

</Tab>
</Tabs>

### Using a connectivity graph

Use the graph path when a KNN graph, affinity graph, or domain-specific connectivity graph is already available.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/preprocessing/spectral_embedding.hpp>

namespace spectral_embedding = cuvs::preprocessing::spectral_embedding;

raft::device_resources res;
spectral_embedding::params params;
params.n_components = 2;
params.norm_laplacian = true;
params.drop_first = true;

raft::device_coo_matrix_view<float, int, int, int> connectivity_graph =
    load_connectivity_graph();

auto embedding = raft::make_device_matrix<float, int, raft::col_major>(
    res, connectivity_graph.structure_view().get_n_rows(), params.n_components);

spectral_embedding::transform(res,
                              params,
                              connectivity_graph,
                              embedding.view());
```

</Tab>
</Tabs>

### Building a connectivity graph

The helper API can build a connectivity graph from a dense dataset when you want to inspect or reuse the graph before embedding.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/preprocessing/spectral_embedding.hpp>

namespace spectral_embedding = cuvs::preprocessing::spectral_embedding;

spectral_embedding::params params;
params.n_neighbors = 15;

auto connectivity_graph =
    raft::make_device_coo_matrix<float, int, int, int>(
        res, dataset.extent(0), dataset.extent(0));

spectral_embedding::helpers::create_connectivity_graph(
    res, params, dataset, connectivity_graph);
```

</Tab>
</Tabs>

## How Spectral Embedding works

Spectral Embedding turns local neighbor relationships into coordinates. First, it builds or accepts a sparse graph where rows are nodes and edges connect nearby or related rows. Then it builds a graph Laplacian from that connectivity graph. Finally, it computes eigenvectors associated with the smallest eigenvalues and uses those eigenvectors as the embedding coordinates.

The embedding preserves graph structure rather than only preserving global Euclidean variance. This is different from [PCA](/user-guide/api-guides/preprocessing-guide/pca), which is linear and does not build a graph.

## When to use Spectral Embedding

Use Spectral Embedding when clusters or manifolds are connected through local neighborhoods, when a KNN graph is already part of the workflow, or when a lower-dimensional view should preserve local structure.

Spectral Embedding is often useful before graph-based clustering or visualization. It can also be useful for exploratory analysis when PCA is too simple to reveal the structure of the data.

Avoid Spectral Embedding when a linear projection is sufficient, when the graph would be too expensive to build, or when the application needs a transform that can be cheaply applied to new points without rebuilding graph structure.

## Configuration parameters

### Transform parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `n_components` | Required | Number of embedding dimensions to compute. |
| `n_neighbors` | Required | Number of neighbors used when constructing a graph from a dense dataset. Ignored when using a precomputed connectivity graph. |
| `norm_laplacian` | Required | Whether to use the normalized graph Laplacian. Normalized Laplacians often work better for clustering and uneven graph degrees. |
| `drop_first` | Required | Whether to drop the first eigenvector. This is commonly enabled with a normalized Laplacian because the first eigenvector can be uninformative. |
| `tolerance` | `1e-5` | Eigenvalue solver tolerance. Smaller values can improve convergence quality but increase runtime. |
| `seed` | `nullopt` | Optional random seed used for reproducible KNN graph construction and eigensolver initialization. |

## Tuning

Start with `n_components` equal to the number of dimensions needed by the downstream analysis. For clustering workflows, this is often close to the target number of clusters.

Tune `n_neighbors` carefully when building the graph from dense data. Too few neighbors can disconnect related regions. Too many neighbors can blur local structure and increase graph memory.

Use `norm_laplacian=true` for most clustering-oriented workflows. Enable `drop_first=true` when the first eigenvector mostly captures the trivial connected-component structure rather than useful variation.

Tighten `tolerance` when the embedding is unstable or downstream clustering quality changes unexpectedly. Relax it when eigensolver time dominates and the embedding is already good enough.

## Memory footprint

Spectral Embedding memory is dominated by the input matrix, the sparse connectivity graph, the embedding matrix, and eigensolver workspace.

Variables:

- `N`: Number of rows.
- `D`: Number of input features.
- `C`: Number of embedding components, or `n_components`.
- `k`: Number of neighbors, or `n_neighbors`.
- `M`: Number of graph edges.
- `B_x`: Bytes per input or embedding element.
- `B_w`: Bytes per graph edge weight.
- `B_i`: Bytes per graph index value.

### Scratch and maximum rows

The `scratch` term is represented by the Laplacian and eigensolver workspace terms below, plus allocator padding, CUDA library workspaces, graph-construction temporaries, and memory held by the active memory resource. Use `H = 0.30` for dense-data planning. If you can measure a representative smaller run, use:

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


For dense data, substitute `M ≈ N * k` and solve the peak as a linear function of `N`:

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

For a precomputed graph, use the actual edge count `M`. If eigensolver workspace dominates, measure it once at a smaller `N` and scale conservatively.

### Dense-data path

When the graph is built from a dense dataset, a useful edge-count estimate is:

$$
M \approx N \cdot k
$$

The input and graph storage are approximately:

$$
\begin{aligned}
\text{input\_size} &= N \cdot D \cdot B_x \\
\text{graph\_size} &\approx M \cdot (B_w + 2B_i)
\end{aligned}
$$

### Embedding

The embedding stores `C` coordinates per row:

$$
\text{embedding\_size}
  = N \cdot C \cdot B_x
$$

The dense-data transform peak is approximately:

$$
\begin{aligned}
\text{transform\_peak}
  \approx&\ \text{input\_size}
   + \text{graph\_size} \\
  &+ \text{embedding\_size}
   + \text{laplacian\_workspace}
   + \text{eigensolver\_workspace}
\end{aligned}
$$

For large datasets, reduce `n_neighbors` only when the graph remains connected enough for the analysis task. Reducing `n_components` lowers embedding memory, but graph and eigensolver workspace can still dominate.
