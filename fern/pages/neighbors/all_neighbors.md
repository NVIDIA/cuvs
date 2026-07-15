# All-neighbors

All-neighbors builds a k-NN graph for every vector in a dataset. Instead of asking "what are the nearest neighbors for this new query?", it asks "what are the nearest neighbors for every row I already have?"

Use it when the graph itself is the output, for example for clustering, visualization, graph analytics, or algorithms that need a neighborhood graph before doing their own work.

## Example API Usage

[C API](/api-reference/c-api-neighbors-all-neighbors) | [C++ API](/api-reference/cpp-api-neighbors-all-neighbors) | [Python API](/api-reference/python-api-neighbors-all-neighbors)

### Building a graph

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/all_neighbors.h>
#include <cuvs/neighbors/nn_descent.h>

cuvsResources_t res;
cuvsAllNeighborsIndexParams_t params;
DLManagedTensor *dataset;
DLManagedTensor *indices;
DLManagedTensor *distances;

int64_t n_rows = 100000;
int64_t dim = 128;
int64_t k = 32;

// Populate DLPack tensors. Dataset may be host or device float32 data.
load_dataset(dataset);
allocate_device_indices(indices, n_rows, k);
allocate_device_distances(distances, n_rows, k);

cuvsResourcesCreate(&res);
cuvsAllNeighborsIndexParamsCreate(&params);

params->algo = CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT;
params->metric = L2Expanded;
params->n_clusters = 1;
params->overlap_factor = 1;

cuvsNNDescentIndexParamsCreate(&params->nn_descent_params);
params->nn_descent_params->graph_degree = k;
params->nn_descent_params->intermediate_graph_degree = 2 * k;
params->nn_descent_params->max_iterations = 20;

cuvsAllNeighborsBuild(
    res,
    params,
    dataset,
    indices,
    distances,
    NULL,
    1.0f);

// This also destroys params->nn_descent_params when it is set.
cuvsAllNeighborsIndexParamsDestroy(params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/all_neighbors.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_device_dataset();

int64_t n_rows = dataset.extent(0);
int64_t k = 32;

auto indices = raft::make_device_matrix<int64_t, int64_t>(res, n_rows, k);
auto distances = raft::make_device_matrix<float, int64_t>(res, n_rows, k);

all_neighbors::all_neighbors_params params;
params.metric = cuvs::distance::DistanceType::L2Expanded;
params.n_clusters = 1;
params.overlap_factor = 1;

all_neighbors::graph_build_params::nn_descent_params nn_params;
nn_params.graph_degree = k;
nn_params.intermediate_graph_degree = 2 * k;
nn_params.max_iterations = 20;
nn_params.metric = params.metric;
params.graph_build_params = nn_params;

all_neighbors::build(
    res,
    params,
    dataset,
    indices.view(),
    distances.view());
```

</Tab>
<Tab title="Python">

```python
import cupy as cp
from cuvs.neighbors import all_neighbors, nn_descent

dataset = load_data()
k = 32

nn_params = nn_descent.IndexParams(
    metric="sqeuclidean",
    graph_degree=k,
    intermediate_graph_degree=2 * k,
    max_iterations=20,
)

params = all_neighbors.AllNeighborsParams(
    algo="nn_descent",
    metric="sqeuclidean",
    n_clusters=1,
    overlap_factor=1,
    nn_descent_params=nn_params,
)

distances = cp.empty((dataset.shape[0], k), dtype=cp.float32)
indices, distances = all_neighbors.build(
    dataset,
    k,
    params,
    distances=distances,
)
```

</Tab>
</Tabs>

C, C++, and Python expose all-neighbors bindings. Java, Rust, and Go do not currently expose standalone all-neighbors wrappers.

### Building in batches

For host-resident datasets, all-neighbors can partition the data into overlapping clusters. Each row is assigned to `overlap_factor` clusters, local k-NN graphs are built inside each cluster, and the local results are merged into one global graph.

Use batching when the full graph build does not fit comfortably on one GPU, or when you want to distribute work across multiple GPUs.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/all_neighbors.hpp>

using namespace cuvs::neighbors;

raft::device_resources_snmg res;
auto dataset = load_host_dataset();

int64_t n_rows = dataset.extent(0);
int64_t k = 32;

auto indices = raft::make_device_matrix<int64_t, int64_t>(res, n_rows, k);
auto distances = raft::make_device_matrix<float, int64_t>(res, n_rows, k);

all_neighbors::all_neighbors_params params;
params.metric = cuvs::distance::DistanceType::L2Expanded;
params.n_clusters = 8;
params.overlap_factor = 2;
params.graph_build_params =
    all_neighbors::graph_build_params::brute_force_params{};

all_neighbors::build(
    res,
    params,
    dataset,
    indices.view(),
    distances.view());
```

</Tab>
<Tab title="Python">

```python
import cupy as cp
import numpy as np
from cuvs.common.mg_resources import MultiGpuResources
from cuvs.neighbors import all_neighbors

dataset = np.asarray(load_data(), dtype=np.float32)
k = 32

params = all_neighbors.AllNeighborsParams(
    algo="brute_force",
    metric="sqeuclidean",
    n_clusters=8,
    overlap_factor=2,
)

resources = MultiGpuResources()
distances = cp.empty((dataset.shape[0], k), dtype=cp.float32)

indices, distances = all_neighbors.build(
    dataset,
    k,
    params,
    distances=distances,
    resources=resources,
)
```

</Tab>
</Tabs>

Device-resident datasets require `n_clusters = 1`. Put the dataset in host memory when using `n_clusters > 1`.

### Saving graph outputs

All-neighbors does not expose NVIDIA cuVS index serialization or deserialization APIs because the graph arrays are the output. Persist `indices`, `distances`, and `core_distances` if used with your application's tensor, array, or table storage format, then load those arrays into the downstream graph-processing workflow.

### Mutual-reachability distances

All-neighbors can compute mutual-reachability distances for workflows such as robust single linkage or HDBSCAN-style graph construction. Provide both `distances` and `core_distances`; `alpha` controls the mutual-reachability scaling.

<Tabs>
<Tab title="C++">

```cpp
auto core_distances = raft::make_device_vector<float, int64_t>(res, n_rows);

all_neighbors::build(
    res,
    params,
    dataset,
    indices.view(),
    distances.view(),
    core_distances.view(),
    1.0f);
```

</Tab>
<Tab title="Python">

```python
core_distances = cp.empty((dataset.shape[0],), dtype=cp.float32)

indices, distances, core_distances = all_neighbors.build(
    dataset,
    k,
    params,
    distances=distances,
    core_distances=core_distances,
    alpha=1.0,
)
```

</Tab>
</Tabs>

IVF-PQ does not support mutual-reachability distances in all-neighbors. Use brute-force or NN-Descent for that mode.

### Searching an index

All-neighbors does not build a reusable search index. It writes k-NN graph outputs for the input dataset. Use CAGRA, IVF-Flat, IVF-PQ, brute-force, or Vamana when you need to search new query vectors.

## How All-neighbors works

When `n_clusters = 1`, all-neighbors builds one k-NN graph over the whole dataset using the selected local graph builder.

When `n_clusters > 1`, all-neighbors first trains cluster centroids from a sample of the host dataset. Then it assigns each row to the `overlap_factor` nearest clusters. Each cluster builds a local k-NN graph, and NVIDIA cuVS merges those local graphs back into one global graph.

The local graph builder can be brute-force, IVF-PQ, or NN-Descent. Brute-force is exact when `n_clusters = 1`. With batching, even brute-force becomes approximate because each row only compares against rows in its assigned clusters.

## When to use All-neighbors

Use all-neighbors when you need a graph for every vector in the dataset.

Use `n_clusters = 1` when the dataset and chosen local builder fit on one GPU.

Use host data with `n_clusters > 1` when you need batching or multi-GPU execution.

Use brute-force as the local builder when exactness matters and the local batches are small enough.

Use NN-Descent when graph build speed matters and approximate graph quality is acceptable.

Use IVF-PQ when you want a partitioned approximate local builder and do not need mutual-reachability distances.

## Using Filters

All-neighbors does not expose filtered search because it does not search new query vectors. Apply filtering in the downstream graph workflow, or use a search index that supports filters.

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `graph_build_params` | IVF-PQ | Local graph builder used inside each full-dataset or clustered build. The C++ params struct supports brute-force, IVF-PQ, and NN-Descent graph builders. |
| `overlap_factor` | `2` | Number of nearest clusters each row is assigned to during batched host builds. Must be smaller than `n_clusters` when `n_clusters > 1`. |
| `n_clusters` | `1` | Number of clusters or batches. `1` disables batching. Values greater than `1` require host-resident data. |
| `metric` | `L2Expanded` | Distance metric used for graph construction. IVF-PQ supports only `L2Expanded` in all-neighbors. |

## Tuning

Start with the local graph builder. Brute-force gives the clearest baseline, NN-Descent is usually the fastest approximate graph builder, and IVF-PQ can be useful for larger local batches when L2 distance is sufficient.

Set `K` to the number of neighbors required by the downstream graph algorithm. Increasing `K` increases output memory and local graph-builder work.

For batched builds, start with `n_clusters = 4` or `8` and `overlap_factor = 2`. Increase `n_clusters` to reduce per-batch device memory. Increase `overlap_factor` to improve recall, because each row is compared inside more clusters.

Keep the ratio `overlap_factor / n_clusters` in mind. It roughly controls how much of the dataset is active in one batch. A larger ratio usually improves graph quality but uses more device memory and work.

Tune the local builder after the batching shape is stable. For NN-Descent, tune `graph_degree`, `intermediate_graph_degree`, and `max_iterations`. For IVF-PQ, tune the IVF-PQ build/search parameters and refinement rate.

## Memory footprint

All-neighbors memory has three main parts: graph outputs, optional batching metadata, and the temporary memory used by the selected local graph builder. The output graph is always written to device memory. Batched builds also keep managed global merge buffers while local cluster results are merged.

To keep the formulas readable, this section uses short symbols. All estimates are in bytes. The examples convert bytes to MiB by dividing by `1024 * 1024`.

- `N`: Number of database vectors.
- `D`: Vector dimension.
- `K`: Output graph degree, or number of neighbors requested.
- `B`: Bytes per dataset value. All-neighbors currently accepts fp32 data, so `B = 4`.
- `C`: Number of clusters, or `n_clusters`.
- `O`: Cluster overlap, or `overlap_factor`.
- `M`: Maximum cluster size after overlap assignment. A planning estimate is `ceil(N * O / C)`, but real clusters may be uneven.
- `S`: Subsample rows used for centroid training. Approximately `min(N / C, 50000)`, with a small-data fallback up to `5000`.
- `R`: `1` when `distances` is provided, otherwise `0`.
- `Q`: `1` when `core_distances` is provided, otherwise `0`.
- `S_idx`: Bytes per output neighbor ID, currently `sizeof(int64_t)`.

The named terms in the formulas are also memory sizes:

- `indices_size`: Device memory for output neighbor IDs.
- `distances_size`: Device memory for output distances.
- `core_distances_size`: Device memory for mutual-reachability core distances.
- `local_builder_peak`: Peak temporary memory for the chosen local graph builder on at most `M` rows.

### Scratch and maximum vectors

The formulas below include output buffers, batching metadata, and local builder peaks. Additional scratch comes from allocator padding, local graph-builder workspaces, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.25` for all-neighbors builds, then add the scratch/headroom recommended by the chosen local builder. If you can measure a representative smaller run, use:

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


With `n_clusters = 1`, solve the single-cluster formula using the local builder peak for `N` rows. With batched builds, substitute `M = ceil(N * O / C)` and solve:

$$
\text{peak\_without\_scratch}(N)
  =
  N \cdot B_{\text{global\_per\_vector}}
  + M(N) \cdot B_{\text{local\_per\_vector}}
  + B_{\text{fixed}}
$$

In this formula, `M(N) = ceil(N * O / C)` is the estimated largest local cluster size for a candidate dataset size `N`. `B_global_per_vector` is the per-vector memory for global outputs and merge metadata. `B_local_per_vector` is the per-vector memory used by the selected local builder when it processes one local cluster. Use the local builder's own memory formula to estimate `B_local_per_vector`.

Then try increasing `N` until `peak_without_scratch(N) <= M_usable` no longer holds. The largest feasible dataset is the smaller value from device memory and host/managed-memory limits.

### Baseline output memory

The output graph memory is:

$$
\begin{aligned}
\text{indices\_size}
&= N \times K \times S_{\text{idx}}
\end{aligned}
$$

$$
\begin{aligned}
\text{distances\_size}
&= R \times N \times K \times 4
\end{aligned}
$$

$$
\begin{aligned}
\text{core\_distances\_size}
&= Q \times N \times 4
\end{aligned}
$$

The baseline output footprint is:

$$
\begin{aligned}
\text{output\_size}
&= \text{indices\_size} \\
&\quad + \text{distances\_size} \\
&\quad + \text{core\_distances\_size}
\end{aligned}
$$

**Example** (`N = 1e6`, `K = 32`, with distances and no core distances):

- `indices_size = 256000000 B = 244.14 MiB`
- `distances_size = 128000000 B = 122.07 MiB`
- `output_size = 384000000 B = 366.21 MiB`

### Single-cluster build memory

With `n_clusters = 1`, all-neighbors runs the selected local builder over the full dataset. Device-resident input stays on device; host input is copied to a device build buffer by local builders that require device data.

The planning estimate is:

$$
\begin{aligned}
\text{single\_build\_peak}
&\approx N \times D \times B \\
&\quad + \text{output\_size} \\
&\quad + \text{local\_builder\_peak}(N, D, K)
\end{aligned}
$$

For brute-force, `local_builder_peak` is mostly the brute-force query workspace and any temporary distance output if `distances` was not provided.

For NN-Descent, use the NN-Descent guide formulas with `N` rows and output degree `K`.

For IVF-PQ, use the IVF-PQ guide formulas with a candidate count based on the IVF-PQ refinement settings.

### Batched build memory

With `n_clusters > 1`, the dataset must be host-resident. All-neighbors builds local graphs on at most `M` rows at a time and merges each local graph into managed global buffers.

Centroid training and cluster assignment use:

$$
\begin{aligned}
\text{centroid\_training\_peak}
&\approx S \times D \times B \\
&\quad + C \times D \times B
\end{aligned}
$$

$$
\begin{aligned}
\text{assignment\_peak}
&\approx \left\lceil \frac{N}{C} \right\rceil
  \times D \times B \\
&\quad + \left\lceil \frac{N}{C} \right\rceil
  \times O \times (S_{\text{idx}} + 4) \\
&\quad + N \times O \times S_{\text{idx}}
\end{aligned}
$$

The global merge buffers are allocated even if the caller does not request final distances:

$$
\begin{aligned}
\text{global\_merge\_size}
&= N \times K \times (S_{\text{idx}} + 4)
\end{aligned}
$$

Reusable per-cluster buffers include the gathered cluster data, local graph outputs, and merge buffers:

$$
\begin{aligned}
\text{cluster\_buffer\_size}
&\approx M \times D \times B \\
&\quad + M \times K \times (2 \times S_{\text{idx}} + 4) \\
&\quad + M \times S_{\text{idx}}
\end{aligned}
$$

Host batching metadata stores inverted cluster membership:

$$
\begin{aligned}
\text{batch\_metadata\_size}
&\approx N \times O \times S_{\text{idx}} \\
&\quad + 2 \times C \times S_{\text{idx}}
\end{aligned}
$$

The batched peak is approximately:

$$
\begin{aligned}
\text{batch\_build\_peak}
&\approx \text{output\_size} \\
&\quad + \text{global\_merge\_size} \\
&\quad + \text{batch\_metadata\_size} \\
&\quad + \max\!\big(
  \text{centroid\_training\_peak}, \\
&\qquad\qquad
  \text{assignment\_peak}, \\
&\qquad\qquad
  \text{cluster\_buffer\_size} \\
&\qquad\qquad
  + \text{local\_builder\_peak}(M, D, K)
\big)
\end{aligned}
$$

If `core_distances` is provided, all-neighbors computes a first graph to get core distances and then rebuilds in mutual-reachability space. Peak memory is similar, but runtime roughly includes two local graph builds. The extra persistent output is `core_distances_size`.

**Example** (`N = 1e6`, `D = 128`, `K = 32`, `C = 8`, `O = 2`, with distances):

- `M ≈ ceil(1e6 * 2 / 8) = 250000`
- `output_size = 384000000 B = 366.21 MiB`
- `global_merge_size = 384000000 B = 366.21 MiB`
- `batch_metadata_size = 16000128 B = 15.26 MiB`
- `cluster_buffer_size = 224000000 B = 213.62 MiB`

Add the selected local builder peak on `M = 250000` rows to estimate the dominant per-cluster build phase.

### Search memory usage

All-neighbors does not search new query vectors, so it has no query workspace. Downstream search memory depends on the index or graph workflow that consumes the generated k-NN graph.
