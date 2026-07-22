# Vamana

Vamana is a graph-building algorithm used by DiskANN. Think of every vector as a point, and think of the graph as a map that connects each point to nearby points. DiskANN uses that map to search very large indexes efficiently, including indexes that are stored on SSD.

NVIDIA cuVS provides a GPU-accelerated Vamana build path. It builds the graph on the GPU, then serializes it in a format compatible with the open-source [DiskANN](https://github.com/microsoft/DiskANN) library for CPU search.

Vamana works well when you want to build large DiskANN-compatible graph indexes quickly on the GPU and use them in CPU or SSD-backed DiskANN search workflows.

## Example API Usage

[C API](/api-reference/c-api-neighbors-vamana) | [C++ API](/api-reference/cpp-api-neighbors-vamana) | [Python API](/api-reference/python-api-neighbors-vamana) | [Rust API](/api-reference/rust-api-cuvs-neighbors-vamana)

Vamana currently supports build and serialize operations in NVIDIA cuVS. Search is performed by loading the serialized index with DiskANN. Java and Go do not currently expose standalone Vamana bindings.

### Building an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/vamana.h>

cuvsResources_t res;
cuvsVamanaIndexParams_t index_params;
cuvsVamanaIndex_t index;
DLManagedTensor *dataset;

// Populate tensor with row-major data.
load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsVamanaIndexParamsCreate(&index_params);
cuvsVamanaIndexCreate(&index);

index_params->metric = L2Expanded;
index_params->graph_degree = 64;
index_params->visited_size = 128;
index_params->queue_size = 255;

cuvsVamanaBuild(res, index_params, dataset, index);

cuvsVamanaIndexDestroy(index);
cuvsVamanaIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/vamana.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
vamana::index_params index_params;

index_params.graph_degree = 64;
index_params.visited_size = 128;
index_params.queue_size = 255;
index_params.metric = cuvs::distance::DistanceType::L2Expanded;

auto index = vamana::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import vamana

dataset = load_data()
index_params = vamana.IndexParams(
    metric="sqeuclidean",
    graph_degree=64,
    visited_size=128,
    queue_size=255,
)

index = vamana.build(index_params, dataset)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::vamana::{Index, IndexParams};
use cuvs::{ManagedTensor, Resources, Result};

fn build_vamana_index(dataset: &ndarray::Array2<f32>) -> Result<Index> {
    let res = Resources::new()?;
    let dataset = ManagedTensor::from(dataset).to_device(&res)?;
    let index_params = IndexParams::new()?
        .set_graph_degree(64)
        .set_visited_size(128)
        .set_queue_size(255);

    Index::build(&res, &index_params, dataset)
}
```

</Tab>
</Tabs>

### Serializing an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/vamana.h>

cuvsResources_t res;
cuvsVamanaIndexParams_t index_params;
cuvsVamanaIndex_t index;
DLManagedTensor *dataset;

cuvsResourcesCreate(&res);
cuvsVamanaIndexParamsCreate(&index_params);
cuvsVamanaIndexCreate(&index);

// ... build index from dataset ...
cuvsVamanaBuild(res, index_params, dataset, index);

// Writes DiskANN-compatible files using this path prefix.
cuvsVamanaSerialize(res, "/tmp/cuvs-vamana/index", index, true);

cuvsVamanaIndexDestroy(index);
cuvsVamanaIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/vamana.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
vamana::index_params index_params;

auto index = vamana::build(res, index_params, dataset);

// Writes DiskANN-compatible files using this path prefix.
vamana::serialize(res, "/tmp/cuvs-vamana/index", index, true);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import vamana

dataset = load_data()
index = vamana.build(vamana.IndexParams(), dataset)

# Writes DiskANN-compatible files using this path prefix.
vamana.save("/tmp/cuvs-vamana/index", index, include_dataset=True)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::vamana::{Index, IndexParams};
use cuvs::{ManagedTensor, Resources, Result};

fn build_and_save_vamana_index(dataset: &ndarray::Array2<f32>) -> Result<()> {
    let res = Resources::new()?;
    let dataset = ManagedTensor::from(dataset).to_device(&res)?;
    let index_params = IndexParams::new()?;
    let index = Index::build(&res, &index_params, dataset)?;

    // Writes DiskANN-compatible files using this path prefix.
    index.serialize(&res, "/tmp/cuvs-vamana/index", true)
}
```

</Tab>
</Tabs>

### Loading a serialized index

NVIDIA cuVS Vamana writes DiskANN-compatible files but does not currently expose a Vamana deserialization or search API. Load the serialized output with DiskANN or another DiskANN-compatible search layer.

## How Vamana works

Vamana builds a directed graph over the dataset. Each vector keeps up to `graph_degree` outgoing edges to other vectors.

At a high level, the builder:

1. Starts from a medoid vector, which is a central entry point into the graph.
2. Inserts vectors in batches.
3. Uses graph traversal to find candidate neighbors for each inserted vector.
4. Adds forward and reverse edges.
5. Prunes edges so each vector keeps a compact set of useful neighbors.

The `visited_size` and `queue_size` parameters control how much of the graph each insertion can explore. The `alpha` parameter controls pruning aggressiveness.

## When to use Vamana

Use Vamana when you need a DiskANN-compatible graph and want GPU-accelerated build.

Vamana is a good fit for very large datasets, SSD-backed search workflows, and hybrid workflows where a GPU-built graph is converted to CPU for DiskANN search.

Avoid Vamana when you need an in-NVIDIA cuVS search API today. In that case, use CAGRA for GPU graph search, or use the serialized Vamana output with DiskANN for CPU search.

## Interoperability with CPU DiskANN

The Vamana serialize APIs write files in a DiskANN-compatible format. This lets NVIDIA cuVS build the graph quickly on the GPU, then lets DiskANN load the serialized files for CPU search.

Set `include_dataset = true` when the serialized index should include the dataset. Set it to `false` when the dataset is already available in the format expected by the downstream DiskANN workflow, or when writing a sector-aligned index with externally prepared quantized data.

The C++ API also supports loading DiskANN-style PQ codebooks before build:

```cpp
#include <cuvs/neighbors/vamana.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
vamana::index_params index_params;

index_params.codebooks =
    vamana::deserialize_codebooks("/tmp/diskann/pivots", dataset.extent(1));

auto index = vamana::build(res, index_params, dataset);
vamana::serialize(
    res,
    "/tmp/cuvs-vamana/index",
    index,
    false,
    true);
```

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / `sqeuclidean` | Distance metric used during graph construction. Vamana currently supports L2-style metrics. |
| `metric_arg` | `2.0` | Extra argument for metrics that need one, such as Minkowski distance. |
| `graph_degree` | `32` | Maximum number of outgoing graph edges kept per vector. This corresponds to the `R` parameter in Vamana literature. |
| `visited_size` | `64` | Maximum number of visited nodes saved during each graph traversal. This loosely corresponds to the `L` parameter in Vamana literature. |
| `vamana_iters` | `1.0` | Number of full insertion passes over the dataset. More passes can improve graph quality, but increase build time. |
| `alpha` | `1.2` | Pruning parameter. Larger values usually prune less aggressively and can improve recall at higher graph cost. |
| `max_fraction` | `0.06` | Maximum fraction of the dataset inserted in one batch. Larger batches can build faster but may reduce graph quality. |
| `batch_base` | `2.0` | Growth factor for insertion batch sizes. |
| `queue_size` | `127` | Candidate queue size used during graph traversal. This should be one less than a power of two and should be larger than `visited_size`. |
| `reverse_batchsize` | `1000000` | Maximum reverse-edge processing batch size. Lower values can reduce temporary memory during reverse-edge processing. |
| `codebooks` | None | Optional C++ PQ codebook parameters loaded with `deserialize_codebooks` for DiskANN-compatible quantized output workflows. |

## Tuning

Tune `graph_degree` first. Larger values give each vector more outgoing edges, which can improve downstream DiskANN recall but increases graph memory, build time, and serialized index size.

Tune `visited_size` and `queue_size` together. Larger values let insertion searches explore more of the graph, which can improve graph quality but increases temporary memory and build cost. Keep `queue_size` larger than `visited_size`.

Increase `vamana_iters` when graph quality matters more than build time. Each extra iteration reinserts all vectors, so build time can increase substantially.

Use `max_fraction`, `batch_base`, and `reverse_batchsize` to manage build speed and temporary memory. Larger insertion batches are faster but can reduce graph quality. Smaller reverse-edge batches reduce peak memory at the cost of more processing chunks.

## Memory footprint

Vamana memory has two main parts: the dataset and the graph. During build, the dataset must be available to the GPU. The final graph stores up to `graph_degree` neighbor IDs for every vector.

To keep the formulas readable, this section uses short symbols. All estimates are in bytes. The examples convert bytes to MiB by dividing by `1024 * 1024`.

- `N`: Number of database vectors, or rows in the dataset being indexed.
- `D`: Vector dimension, or number of values in each vector.
- `B`: Bytes stored for each vector value. Use `4` for fp32, `1` for int8 and uint8, or the byte width of the dataset representation.
- `G`: Final graph degree. This is the `graph_degree` build parameter.
- `V`: Number of visited nodes kept during traversal. This is the `visited_size` build parameter.
- `Q_c`: Candidate queue size. This is the `queue_size` build parameter.
- `F`: Maximum insertion batch fraction. This is the `max_fraction` build parameter.
- `R_b`: Reverse-edge processing batch size. This is the `reverse_batchsize` build parameter.
- `S_idx`: Bytes per graph neighbor ID. Vamana uses 32-bit graph IDs, so this is usually `4`.

The named terms in the formulas are also memory sizes:

- `dataset_size`: Device memory used by vectors during build.
- `graph_size`: Device memory used by final graph neighbor IDs.
- `build_batch_size`: Maximum number of vectors inserted in one batch.
- `reverse_batch_size`: Maximum number of vectors processed in one reverse-edge batch.
- `insertion_scratch_size`: Temporary memory used while inserting one batch.
- `reverse_scratch_size`: Temporary memory used while processing reverse edges.
- `build_peak`: Approximate peak device memory during build.
- `host_serialize_size`: Host memory used while serializing.
- `metadata_size`: Small fixed index metadata. It is usually negligible compared with the dataset and graph, but include a measured value if exact accounting is required.

### Scratch and maximum vectors

The formulas below include the major insertion and reverse-edge scratch buffers. Additional scratch comes from allocator padding, graph update temporaries, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.25` for build estimates. If you can measure a representative smaller run, use:

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


For fixed `D`, `G`, `V`, `Q_c`, `F`, and `R_b`, the build peak is mostly linear in `N` until `reverse_batch_size` reaches `R_b`. Solve the full `max(...)` expression or use the linear shortcut in the active region:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_vector}}}
  \right\rfloor
$$

If `reverse_batch_size = min(N, R_b)` is capped, treat the reverse scratch term as fixed after `N >= R_b`.

### Baseline memory after build

The baseline device footprint after index construction is:

$$
\begin{aligned}
\text{dataset\_size}
&= N \times D \times B
\end{aligned}
$$

$$
\begin{aligned}
\text{graph\_size}
&= N \times G \times S_{\text{idx}}
\end{aligned}
$$

$$
\begin{aligned}
\text{index\_size}
&\approx
  \text{dataset\_size} \\
&\quad + \text{graph\_size} \\
&\quad + \text{metadata\_size}
\end{aligned}
$$

**Example** (`N = 1e6`, `D = 128`, fp32, `graph_degree = 64`, `IdxT = uint32`):

- `dataset_size = 512,000,000 B = 488.28 MiB`
- `graph_size = 256,000,000 B = 244.14 MiB`
- `index_size ~= 732.42 MiB + small metadata`

### Build peak memory usage

Vamana build inserts vectors in batches and also creates reverse edges. These temporary buffers are affected by `max_fraction`, `queue_size`, `visited_size`, `graph_degree`, and `reverse_batchsize`.

The maximum insertion batch size is:

$$
\begin{aligned}
\text{build\_batch\_size}
&=
\left\lceil F \times N \right\rceil
\end{aligned}
$$

The reverse-edge processing batch size is:

$$
\begin{aligned}
\text{reverse\_batch\_size}
&=
\min(N, R_b)
\end{aligned}
$$

A practical insertion scratch estimate is:

$$
\begin{aligned}
\text{insertion\_scratch\_size}
&\approx
\text{build\_batch\_size}
\times (Q_c + V + G)
\times S_{\text{idx}}
\end{aligned}
$$

A practical reverse-edge scratch estimate is:

$$
\begin{aligned}
\text{reverse\_scratch\_size}
&\approx
\text{reverse\_batch\_size}
\times (2 + G)
\times S_{\text{idx}}
\end{aligned}
$$

The approximate build peak is:

$$
\begin{aligned}
\text{build\_peak}
&\approx
  \text{dataset\_size} \\
&\quad + \text{graph\_size} \\
&\quad + \max\!\big(
  \text{insertion\_scratch\_size}, \\
&\qquad\qquad
  \text{reverse\_scratch\_size}
\big)
\end{aligned}
$$

**Example** (`N = 1e6`, `D = 128`, fp32, `G = 64`, `V = 128`, `Q_c = 255`, `F = 0.06`, `R_b = 1e6`, `S_idx = 4`):

- `build_batch_size = 60,000`
- `insertion_scratch_size = 107,280,000 B = 102.31 MiB`
- `reverse_scratch_size = 264,000,000 B = 251.77 MiB`
- `build_peak ~= 984.19 MiB`

Lower `reverse_batchsize` if reverse-edge processing dominates peak memory. Lower `max_fraction`, `visited_size`, or `queue_size` if insertion scratch dominates.

### Serialization memory usage

Serialization writes a DiskANN-compatible index. The graph must be available on the host while writing. If `include_dataset = true`, the dataset is also included in the serialized output.

$$
\begin{aligned}
\text{host\_graph\_size}
&= N \times G \times S_{\text{idx}}
\end{aligned}
$$

$$
\begin{aligned}
\text{host\_dataset\_size}
&=
\begin{cases}
N \times D \times B, & \text{include dataset} \\
0, & \text{do not include dataset}
\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
\text{host\_serialize\_size}
&\approx
  \text{host\_graph\_size} \\
&\quad + \text{host\_dataset\_size}
\end{aligned}
$$

The serialized files can be smaller than the fixed-degree in-memory graph because DiskANN-style output can store a compact graph representation.
