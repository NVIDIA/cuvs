# Single-linkage

Single-linkage is a hierarchical clustering algorithm. It builds a tree of merges by repeatedly connecting the two closest clusters, where cluster distance is defined by the closest pair of points across the two clusters.

Use single-linkage when you want a dendrogram, want to cut a hierarchy into a chosen number of clusters, or need clustering that can capture chain-like structure. Unlike K-Means, single-linkage does not learn centroids.

## Example API Usage

[C++ API](/api-reference/cpp-api-cluster-agglomerative)

### Clustering data

The C++ API can build labels directly from a dense row-major device matrix. The output dendrogram has `n_rows - 1` rows and two columns, and the labels vector has one label per input row.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/cluster/agglomerative.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace agglomerative = cuvs::cluster::agglomerative;

raft::device_resources res;
raft::device_matrix_view<const float, int, raft::row_major> dataset =
    load_dataset();

auto dendrogram = raft::make_device_matrix<int, int>(
    res, dataset.extent(0) - 1, 2);
auto labels = raft::make_device_vector<int, int>(res, dataset.extent(0));

agglomerative::single_linkage(res,
                              dataset,
                              dendrogram.view(),
                              labels.view(),
                              cuvs::distance::DistanceType::L2Expanded,
                              16,
                              agglomerative::Linkage::KNN_GRAPH,
                              15);
```

</Tab>
</Tabs>

### Building a linkage

Use the helper API when you need the minimum spanning tree, merge distances, or cluster sizes in addition to the dendrogram.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/cluster/agglomerative.hpp>

namespace agglomerative = cuvs::cluster::agglomerative;
namespace linkage_params = agglomerative::helpers::linkage_graph_params;

linkage_params::distance_params params;
params.dist_type = agglomerative::Linkage::KNN_GRAPH;
params.c = 15;

agglomerative::helpers::build_linkage(res,
                                      dataset,
                                      params,
                                      cuvs::distance::DistanceType::L2Expanded,
                                      mst.view(),
                                      dendrogram.view(),
                                      distances.view(),
                                      sizes.view(),
                                      std::nullopt);
```

</Tab>
</Tabs>

## How Single-linkage works

Single-linkage first builds a graph over the data and then finds a minimum spanning tree. Sorting the tree edges from shortest to longest gives the merge order for the dendrogram. Cutting that tree at the right number of components gives cluster labels.

The `PAIRWISE` mode builds from pairwise distances. It is simple and fast for smaller datasets, but its memory grows quadratically. The `KNN_GRAPH` mode builds a sparse nearest-neighbor graph and can scale to much larger datasets, at the cost of more graph construction work.

## When to use Single-linkage

Use single-linkage when the hierarchy itself matters, when clusters may have elongated or connected shapes, or when you need a dendrogram for later analysis.

Avoid single-linkage when small bridges between groups should not merge them. Because it uses the closest pair of points, it can chain through sparse connections that other clustering methods would separate.

## Configuration parameters

### Clustering parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `metric` | Required | Distance metric used to compare rows when constructing the graph. |
| `n_clusters` | Required | Number of clusters to assign after cutting the hierarchy. |
| `linkage` | `KNN_GRAPH` | Graph construction strategy. `PAIRWISE` uses more memory and can be faster for smaller datasets. `KNN_GRAPH` uses a sparse graph for larger datasets. |
| `c` | `15` | Constant used to choose the KNN graph size. Larger values add more graph edges and can improve connectivity, but increase memory and build time. |

## Tuning

Start with `KNN_GRAPH` for large datasets and `PAIRWISE` for smaller datasets where the pairwise distance matrix fits comfortably in memory.

Increase `c` if the KNN graph is too sparse or produces unstable connectivity. Decrease it when memory use or graph construction time is too high.

Choose the distance metric to match the data representation. The hierarchy can change substantially when the metric changes.

## Memory footprint

Single-linkage memory depends strongly on the graph construction mode.

Variables:

- `N`: Number of rows.
- `D`: Number of features per row.
- `k`: Number of neighbors used by the sparse graph.
- `B_x`: Bytes per input element.
- `B_d`: Bytes per distance value.
- `B_i`: Bytes per index value.

### Scratch and maximum rows

The formulas below show the large persistent arrays and graph buffers. `scratch` covers temporary graph-construction buffers, sort or merge workspace, allocator padding, CUDA library workspaces, and memory held by the active memory resource. For planning, reserve a headroom factor `H = 0.30` for single-linkage builds. If you can measure a representative smaller run, use:

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


For KNN graph mode, substitute `k` and rewrite the peak as a linear function of `N`:

$$
\text{peak\_without\_scratch}(N)
  = N \cdot B_{\text{per\_row}} + B_{\text{fixed}}
$$

Then estimate:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_row}}}
  \right\rfloor
$$

For pairwise mode, the dense distance matrix is quadratic. If both inputs have `N` rows, solve:

$$
N_{\max}
  \approx
  \left\lfloor
    \sqrt{\frac{M_{\text{usable}} - B_{\text{fixed}}}{B_d}}
  \right\rfloor
$$

### Pairwise mode

The pairwise path can require a dense distance matrix:

$$
\text{pairwise\_distance\_size}
  \approx N^2 \cdot B_d
$$

This mode is usually only practical when `N` is small enough for the quadratic matrix to fit comfortably.

### KNN graph mode

The sparse graph path stores roughly `N * k` weighted edges:

$$
\text{knn\_graph\_size}
  \approx N \cdot k \cdot (B_d + 2B_i)
$$

The dendrogram and labels are smaller:

$$
\begin{aligned}
\text{dendrogram\_size} &\approx 2(N - 1) \cdot B_i \\
\text{labels\_size} &= N \cdot B_i
\end{aligned}
$$

The sparse path peak is approximately:

$$
\begin{aligned}
\text{knn\_peak}
  \approx&\ N \cdot D \cdot B_x
   + \text{knn\_graph\_size} \\
  &+ \text{dendrogram\_size}
   + \text{labels\_size}
   + \text{scratch}
\end{aligned}
$$
