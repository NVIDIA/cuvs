# NN-Descent

NN-Descent is a GPU-accelerated graph builder for approximate all-neighbors k-NN graphs. It starts with a rough random graph, repeatedly asks each vector to check its neighbors' neighbors, and keeps better edges as it finds them.

NN-Descent is useful when you need a k-NN graph as an output, or when another NVIDIA cuVS index such as CAGRA needs an approximate graph as an input. It is not a query-time search index for new query vectors.

## Example API Usage

[C API](/api-reference/c-api-neighbors-nn-descent) | [C++ API](/api-reference/cpp-api-neighbors-nn-descent) | [Python API](/api-reference/python-api-neighbors-nn-descent)

### Building a graph

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/nn_descent.h>

cuvsResources_t res;
cuvsNNDescentIndexParams_t index_params;
cuvsNNDescentIndex_t index;
DLManagedTensor *dataset;

// Populate the DLPack tensor with host or device data.
load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsNNDescentIndexParamsCreate(&index_params);
cuvsNNDescentIndexCreate(&index);

index_params->metric = L2Expanded;
index_params->graph_degree = 64;
index_params->intermediate_graph_degree = 128;
index_params->max_iterations = 20;
index_params->termination_threshold = 0.0001f;
index_params->return_distances = true;
index_params->dist_comp_dtype = NND_DIST_COMP_AUTO;

cuvsNNDescentBuild(res, index_params, dataset, NULL, index);

cuvsNNDescentIndexDestroy(index);
cuvsNNDescentIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/nn_descent.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_device_dataset();

nn_descent::index_params index_params;
index_params.metric = cuvs::distance::DistanceType::L2Expanded;
index_params.graph_degree = 64;
index_params.intermediate_graph_degree = 128;
index_params.max_iterations = 20;
index_params.termination_threshold = 0.0001f;
index_params.return_distances = true;
index_params.dist_comp_dtype = nn_descent::DIST_COMP_DTYPE::AUTO;

auto index = nn_descent::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import nn_descent

dataset = load_data()

index_params = nn_descent.IndexParams(
    metric="sqeuclidean",
    graph_degree=64,
    intermediate_graph_degree=128,
    max_iterations=20,
    termination_threshold=0.0001,
    return_distances=True,
    dist_comp_dtype="auto",
)

index = nn_descent.build(index_params, dataset)
```

</Tab>
</Tabs>

C, C++, and Python expose standalone NN-Descent bindings. Java, Rust, and Go do not currently expose standalone NN-Descent wrappers, although those bindings may expose CAGRA settings that use NN-Descent internally.

### Reading the graph

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/nn_descent.h>

cuvsResources_t res;
cuvsNNDescentIndex_t index;
DLManagedTensor *graph;
DLManagedTensor *distances;

// ... build the index ...
allocate_host_graph(graph, n_rows, graph_degree);
allocate_host_distances(distances, n_rows, graph_degree);

cuvsNNDescentIndexGetGraph(res, index, graph);
cuvsNNDescentIndexGetDistances(res, index, distances);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/nn_descent.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_device_dataset();

nn_descent::index_params index_params;
index_params.return_distances = true;

auto index = nn_descent::build(res, index_params, dataset);

auto graph = index.graph();
auto distances = index.distances();

if (distances.has_value()) {
  use_graph_and_distances(graph, distances.value());
}
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import nn_descent

dataset = load_data()
index_params = nn_descent.IndexParams(return_distances=True)

index = nn_descent.build(index_params, dataset)

graph = index.graph
distances = index.distances
```

</Tab>
</Tabs>

Distances are available only when `return_distances` is enabled during build. If you provide a preallocated output graph, disable `return_distances` unless you also use a binding that can provide a matching distance output.

### Searching an index

NN-Descent builds a graph over the input dataset. It does not expose an NVIDIA cuVS search API for new query vectors. Use CAGRA, IVF-Flat, IVF-PQ, brute-force, or Vamana when you need query-time search.

### Saving graph outputs

NN-Descent does not expose NVIDIA cuVS index serialization or deserialization APIs because its output is a graph, not a query-time search index. Persist the returned graph and optional distances with your application's tensor, array, or table storage format, then load those arrays into the downstream workflow that consumes the graph.

## How NN-Descent works

NN-Descent starts with a random neighbor graph. Each vector has a list of candidate neighbors, but those candidates are only a first guess.

During each iteration, the algorithm samples new and old neighbors, checks pairs of vectors that are likely to become better neighbors, and updates the graph when it finds closer candidates.

The process stops when `max_iterations` is reached or when the number of graph updates falls below `termination_threshold`.

The final result is an all-neighbors graph with `graph_degree` neighbors per vector.

## When to use NN-Descent

Use NN-Descent when you need an approximate k-NN graph rather than a searchable index.

Use NN-Descent as a graph builder when CAGRA or all-neighbors needs a fast approximate initial graph.

Use brute-force all-neighbors when exact graph quality matters more than build speed.

Use CAGRA, IVF-Flat, IVF-PQ, or Vamana when the main task is searching new query vectors.

## Interoperability with CAGRA and all-neighbors

CAGRA can use NN-Descent to build its initial graph before pruning. This is useful when you want CAGRA search behavior but prefer NN-Descent graph construction.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_device_dataset();

cagra::index_params index_params;
index_params.graph_degree = 64;
index_params.intermediate_graph_degree = 128;
index_params.graph_build_params =
    cagra::graph_build_params::nn_descent_params(
        index_params.intermediate_graph_degree,
        index_params.metric);

auto index = cagra::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import cagra

dataset = load_data()

index_params = cagra.IndexParams(
    build_algo="nn_descent",
    graph_degree=64,
    intermediate_graph_degree=128,
    nn_descent_niter=20,
)

index = cagra.build(index_params, dataset)
```

</Tab>
</Tabs>

All-neighbors can also use NN-Descent as its local graph builder.

## Using Filters

NN-Descent does not expose filtered search because it does not expose search. Apply filtering in the downstream search index or graph-processing workflow that consumes the generated graph.

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / `sqeuclidean` | Distance metric used while building the graph. Supported metrics include L2, squared L2, cosine, inner product, L1, and bitwise Hamming for int8 or uint8 data. |
| `metric_arg` | `2.0` | Extra argument for metrics that need one, such as Minkowski distance. |
| `graph_degree` | `64` | Number of neighbors kept in the final output graph for each vector. |
| `intermediate_graph_degree` | `128` | Larger internal graph degree used while refining candidate neighbors. It is usually at least `1.5 * graph_degree`. |
| `max_iterations` | `20` | Maximum number of graph-refinement iterations. More iterations can improve graph quality, but increase build time. |
| `termination_threshold` | `0.0001` | Early-stop threshold based on how many graph updates are still happening. Smaller values can run longer. |
| `return_distances` | `true` | Stores distances for the returned graph. Disable this to reduce memory when only neighbor IDs are needed. |
| `dist_comp_dtype` | `AUTO` / `"auto"` | Distance-computation dtype. `AUTO` chooses from the dataset shape, `FP32` favors precision, and `FP16` favors speed and memory use. |

## Tuning

Start with `graph_degree`. It controls the size of the final graph and should match the number of neighbors your downstream workflow needs.

Increase `intermediate_graph_degree` when the final graph quality is too low. This gives NN-Descent more candidates to consider before it trims the graph down to `graph_degree`.

Increase `max_iterations` when graph quality is still improving at the end of the build. Lower it when build time matters more than graph quality.

Tune `termination_threshold` after choosing the degrees. A larger threshold stops earlier; a smaller threshold keeps refining until updates become rarer.

Use `dist_comp_dtype="fp16"` or `DIST_COMP_DTYPE::FP16` when speed and memory matter more than distance precision. Use fp32 when small numeric differences matter.

## Memory footprint

NN-Descent memory has two parts: the graph returned by the index and temporary graph-building workspaces. The final index does not keep the original dataset attached, but build needs a converted copy of the dataset on the GPU.

To keep the formulas readable, this section uses short symbols. All estimates are in bytes. The examples convert bytes to MiB by dividing by `1024 * 1024`.

- `N`: Number of database vectors.
- `D`: Vector dimension.
- `B_in`: Bytes per input vector value, such as `4` for fp32 or `2` for fp16.
- `B_comp`: Bytes per stored compute value. Use `4` for fp32 distance computation and `2` for fp16 distance computation.
- `G`: Final graph degree, or `graph_degree`, after clipping to the dataset size.
- `I`: Intermediate graph degree, or `intermediate_graph_degree`, after clipping to the dataset size.
- `E_G`: Expanded graph degree, `roundUp32(G)` when `G <= 32`, otherwise `roundUp32(1.3 * G)`.
- `E_I`: Expanded intermediate degree, `roundUp32(I)` when `I <= 32`, otherwise `roundUp32(1.3 * I)`.
- `S`: Sample width used by the implementation, currently `32`.
- `R`: `1` when `return_distances` is enabled, otherwise `0`.
- `M_l2`: `1` for L2 metrics that store norms, otherwise `0`.
- `S_idx`: Bytes per graph ID, currently `sizeof(uint32_t)`.

The named terms in the formulas are also memory sizes:

- `graph_size`: Host memory for the returned neighbor IDs.
- `distances_size`: Device memory for returned distances when requested.
- `compute_dataset_size`: Device memory for the converted build dataset.
- `*_peak`: Temporary peak memory for build workspaces.

### Scratch and maximum vectors

The formulas below already include the major NN-Descent workspaces. Additional scratch comes from allocator padding, CUDA library workspaces, memory-resource pools, and small implementation buffers. Use `H = 0.30` for build estimates. If you can measure a representative smaller run, use:

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


For fixed `D`, `G`, `I`, and dtype settings, the device and host peaks are linear in `N`. Rewrite the selected peak as:

$$
\text{peak\_without\_scratch}(N)
  = N \cdot B_{\text{per\_vector}} + B_{\text{fixed}}
$$

and solve:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_vector}}}
  \right\rfloor
$$

Check device and host memory separately. The usable `N` is the smaller of the device-memory limit and the host/pinned-memory limit.

### Baseline memory after build

The baseline host memory kept by the NN-Descent index is:

$$
\begin{aligned}
\text{graph\_size}
&= N \times G \times S_{\text{idx}}
\end{aligned}
$$

If `return_distances` is enabled, the index also keeps graph distances on device:

$$
\begin{aligned}
\text{distances\_size}
&= R \times N \times G \times 4
\end{aligned}
$$

The total index footprint is approximately:

$$
\begin{aligned}
\text{index\_size}
&\approx \text{graph\_size} \\
&\quad + \text{distances\_size}
\end{aligned}
$$

**Example** (`N = 1e6`, `graph_degree = 64`, `return_distances = true`):

- `graph_size = 256000000 B = 244.14 MiB`
- `distances_size = 256000000 B = 244.14 MiB`
- `index_size = 512000000 B = 488.28 MiB`

### Build peak memory usage

During build, NN-Descent stores a converted dataset on the GPU:

$$
\begin{aligned}
\text{compute\_dataset\_size}
&= N \times D \times B_{\text{comp}}
\end{aligned}
$$

For L2 metrics, it also stores one fp32 norm per vector:

$$
\begin{aligned}
\text{l2\_norms\_size}
&= M_{l2} \times N \times 4
\end{aligned}
$$

The main device workspaces are a sampled graph buffer, a distance buffer, locks, and list-size counters:

$$
\begin{aligned}
\text{device\_workspace\_size}
&\approx N \times S \times S_{\text{idx}} \\
&\quad + N \times S \times 4 \\
&\quad + N \times 4 \\
&\quad + 2 \times N \times 8
\end{aligned}
$$

Using the current sample width and `uint32_t` IDs, this is:

$$
\begin{aligned}
\text{device\_workspace\_size}
&\approx N \times 276
\end{aligned}
$$

The approximate device peak is:

$$
\begin{aligned}
\text{device\_build\_peak}
&\approx N \times D \times B_{\text{in}} \\
&\quad + \text{compute\_dataset\_size} \\
&\quad + \text{l2\_norms\_size} \\
&\quad + \text{device\_workspace\_size} \\
&\quad + \text{distances\_size}
\end{aligned}
$$

If the input dataset is already on the host, the first term is not device-resident input data. The build still keeps the converted compute dataset on device.

Host and pinned memory include the returned graph, the expanded internal graph, graph distances, sample buffers, reverse-edge buffers, and a Bloom filter used for sampling:

$$
\begin{aligned}
\text{host\_workspace\_size}
&\approx \text{graph\_size} \\
&\quad + N \times E_G
  \times (S_{\text{idx}} + 4) \\
&\quad + N \times E_I \times 2 \\
&\quad + N \times 912 \\
&\quad + R \times N \times G \times 4
\end{aligned}
$$

The `N * 912` term comes from fixed-width sample, reverse-edge, update, and list-size buffers. The `N * E_I * 2` term estimates the Bloom filter bitset.

**Example** (`N = 1e6`, `D = 128`, fp32 input, fp16 compute, `G = 64`, `I = 128`, `return_distances = true`):

- `compute_dataset_size = 256000000 B = 244.14 MiB`
- `device_workspace_size = 276000000 B = 263.21 MiB`
- `distances_size = 256000000 B = 244.14 MiB`
- `device_build_peak = 1304000000 B = 1243.59 MiB`

With `G = 64`, `E_G = 96`; with `I = 128`, `E_I = 192`:

- `host_workspace_size = 2320000000 B = 2212.52 MiB`

### Search memory usage

NN-Descent does not search new query vectors, so it has no query workspace. Downstream search memory depends on the index or workflow that consumes the generated graph.
