# Spectral Clustering

Spectral clustering groups data by first building a graph, then using eigenvectors of that graph to create a clustering-friendly embedding. It is useful when clusters are connected by graph structure rather than being compact around centroids.

Use spectral clustering when the shape of the data is not well described by spherical clusters, or when you already have a meaningful connectivity graph. It is more expensive than K-Means, but it can separate clusters that K-Means would blend together.

## Example API Usage

[C++ API](/api-reference/cpp-api-cluster-spectral)

### Clustering a dense dataset

The dense-data overload builds a KNN connectivity graph internally, computes the spectral embedding, and assigns labels.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/cluster/spectral.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>

namespace spectral = cuvs::cluster::spectral;

raft::device_resources res;
raft::device_matrix_view<float, int, raft::row_major> dataset = load_dataset();

spectral::params params;
params.n_clusters = 8;
params.n_components = 8;
params.n_neighbors = 15;
params.n_init = 10;
params.tolerance = 1e-5f;
params.rng_state = raft::random::RngState{1234};

auto labels = raft::make_device_vector<int, int>(res, dataset.extent(0));

spectral::fit_predict(res, params, dataset, labels.view());
```

</Tab>
</Tabs>

### Clustering a connectivity graph

Use the graph overload when the connectivity graph is already available or when you want to control graph construction separately.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/cluster/spectral.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>

namespace spectral = cuvs::cluster::spectral;
namespace embedding = cuvs::preprocessing::spectral_embedding;

embedding::params graph_params;
graph_params.n_neighbors = 15;

auto graph = raft::make_device_coo_matrix<float, int, int, int>(
    res, n_samples, n_samples, n_edges);

embedding::helpers::create_connectivity_graph(
    res, graph_params, dataset, graph.view());

auto labels = raft::make_device_vector<int, int>(res, n_samples);

spectral::fit_predict(res, params, graph.view(), labels.view());
```

</Tab>
</Tabs>

## How Spectral Clustering works

Spectral clustering builds a graph where nearby rows are connected. It then computes eigenvectors from that graph and uses those eigenvectors as a lower-dimensional embedding. K-Means is applied to the embedding to produce the final labels.

This means spectral clustering depends on both graph quality and embedding quality. A good graph keeps points in the same natural group connected while avoiding too many edges between different groups.

## When to use Spectral Clustering

Use spectral clustering when clusters have curved, connected, or manifold-like structure that centroid methods struggle to capture. It is also useful when a domain-specific connectivity graph already exists.

Avoid spectral clustering when a simple centroid model is enough. The graph and eigenvector computations add memory and runtime overhead compared with K-Means.

## Configuration parameters

### Fit parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `n_clusters` | Required | Number of output clusters. |
| `n_components` | Required | Number of eigenvectors used for the spectral embedding. This is usually equal to `n_clusters`. |
| `n_init` | Required | Number of K-Means initializations used when clustering the embedding. |
| `n_neighbors` | Required | Number of neighbors used when constructing the connectivity graph from dense data. |
| `tolerance` | Required | Tolerance for the eigenvalue solver. |
| `rng_state` | `0` | Random number generator state used for reproducible K-Means initialization. |

## Tuning

Start with `n_components = n_clusters`. Increase `n_components` only when extra embedding dimensions improve downstream clustering quality enough to justify the added work.

Tune `n_neighbors` carefully. Too few neighbors can disconnect natural clusters; too many neighbors can blur the boundaries between clusters.

Increase `n_init` when labels vary noticeably between runs. Lower it when runtime is more important and the embedding is stable.

Use a stricter `tolerance` when the embedding quality is unstable. Relax it when eigensolver time dominates and clustering quality is already sufficient.

## Memory footprint

Spectral clustering memory is dominated by the input data, the connectivity graph, the spectral embedding, and K-Means scratch space on the embedding.

Variables:

- `N`: Number of rows.
- `D`: Number of features per row.
- `K`: Number of clusters.
- `C`: Number of spectral components.
- `M`: Number of graph edges.
- `B_x`: Bytes per input or embedding element.
- `B_w`: Bytes per graph edge weight.
- `B_i`: Bytes per index or label value.

### Scratch and maximum rows

The `scratch` term covers connectivity-graph construction workspace, Laplacian workspace, eigensolver workspace, K-Means workspace on the embedding, allocator padding, and memory held by the active memory resource. For a first estimate, reserve `H = 0.30` because graph and eigensolver workspace can be significant. If you can measure a representative smaller run, use:

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


With dense input, substitute `M ≈ N * n_neighbors` into the graph formula. The peak then becomes approximately linear in `N`:

$$
\text{peak\_without\_scratch}(N)
  = N \cdot B_{\text{per\_row}} + B_{\text{fixed}}
$$

Estimate the maximum row count with:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_row}}}
  \right\rfloor
$$

If the connectivity graph is precomputed, use its actual edge count `M` instead of `N * n_neighbors`.

### Connectivity graph

For dense-data clustering, the graph has roughly `N * n_neighbors` edges:

$$
M \approx N \cdot \text{n\_neighbors}
$$

The sparse COO graph size is approximately:

$$
\text{graph\_size}
  \approx M \cdot (B_w + 2B_i)
$$

### Embedding and labels

The spectral embedding and labels use:

$$
\begin{aligned}
\text{embedding\_size} &= N \cdot C \cdot B_x \\
\text{labels\_size} &= N \cdot B_i
\end{aligned}
$$

The dense-data peak is approximately:

$$
\begin{aligned}
\text{fit\_peak}
  \approx&\ N \cdot D \cdot B_x
   + \text{graph\_size} \\
  &+ \text{embedding\_size}
   + K \cdot C \cdot B_x
   + \text{labels\_size}
   + \text{scratch}
\end{aligned}
$$

For large `N`, reduce `n_neighbors` only if the graph remains connected enough for the clustering task.
