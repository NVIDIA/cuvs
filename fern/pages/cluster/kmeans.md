# K-Means

K-Means is a GPU-accelerated clustering algorithm. It groups rows into `n_clusters` groups by learning one centroid for each group, then assigning every row to the closest centroid.

Use K-Means when you want to summarize a dataset with representative centers, assign vectors to coarse groups, build vector quantizers, or partition data before another algorithm. Unlike an ANN index, K-Means does not build a search structure for nearest-neighbor lookup. Its primary outputs are centroids, labels, inertia, and the number of training iterations.

## Example API Usage

[C API](/api-reference/c-api-cluster-kmeans) | [C++ API](/api-reference/cpp-api-cluster-kmeans) | [Python API](/api-reference/python-api-cluster-kmeans) | [Rust API](/api-reference/rust-api-cuvs-cluster-kmeans)

### Fitting clusters

Fitting learns the cluster centroids from a dataset. The input data can be on the device, and C, C++, and Python also support host-data paths that stream batches to the GPU.

<Tabs>
<Tab title="C">

```c
#include <cuvs/cluster/kmeans.h>
#include <cuvs/core/c_api.h>

cuvsResources_t res;
cuvsKMeansParams_t params;
DLManagedTensor *dataset;
DLManagedTensor *centroids;
double inertia;
int n_iter;

load_dataset(dataset);
allocate_centroids(centroids);

cuvsResourcesCreate(&res);
cuvsKMeansParamsCreate(&params);

params->n_clusters = 1024;
params->max_iter = 300;
params->tol = 1e-4;

cuvsKMeansFit(res, params, dataset, NULL, centroids, &inertia, &n_iter);

cuvsKMeansParamsDestroy(params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <optional>

using namespace cuvs::cluster;

raft::device_resources res;
raft::device_matrix_view<const float, int> dataset = load_dataset();
auto centroids = raft::make_device_matrix<float, int>(res, 1024, dataset.extent(1));

kmeans::params params;
params.n_clusters = 1024;
params.max_iter = 300;
params.tol = 1e-4;

float inertia;
int n_iter;

kmeans::fit(res,
            params,
            dataset,
            std::nullopt,
            centroids.view(),
            raft::make_host_scalar_view(&inertia),
            raft::make_host_scalar_view(&n_iter));
```

</Tab>
<Tab title="Python">

```python
import cupy as cp

from cuvs.cluster.kmeans import KMeansParams, fit

dataset = cp.asarray(load_dataset(), dtype=cp.float32)
params = KMeansParams(n_clusters=1024, max_iter=300, tol=1e-4)

centroids, inertia, n_iter = fit(params, dataset)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::cluster::kmeans::{self, Params};
use cuvs::{ManagedTensor, Resources, Result};

fn fit_kmeans(dataset: &ndarray::Array2<f32>, n_clusters: usize) -> Result<()> {
    let res = Resources::new()?;
    let dataset = ManagedTensor::from(dataset).to_device(&res)?;

    let n_features = dataset.shape()[1];
    let centroids_host = ndarray::Array::<f32, _>::zeros((n_clusters, n_features));
    let mut centroids = ManagedTensor::from(&centroids_host).to_device(&res)?;

    let params = Params::new()?
        .set_n_clusters(n_clusters as i32)
        .set_max_iter(300)
        .set_tol(1e-4);

    let (_inertia, _n_iter) = kmeans::fit(&res, &params, &dataset, &None, &mut centroids)?;
    Ok(())
}
```

</Tab>
</Tabs>

### Fitting clusters on multiple GPUs

Multi-GPU K-Means is exposed through the C++ API. Configure a RAFT handle with NCCL communications, pass each rank's local dataset shard to `fit()`, and NVIDIA cuVS coordinates centroid updates across the participating GPUs.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/cluster/kmeans.hpp>

#include <raft/comms/std_comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>

#include <nccl.h>
#include <optional>

using namespace cuvs::cluster;

void fit_kmeans_multi_gpu(ncclComm_t nccl_comm, int rank, int world_size)
{
  raft::handle_t handle;
  raft::comms::build_comms_nccl_only(&handle, nccl_comm, world_size, rank);

  // Each rank owns one row-major shard of the full dataset on its GPU.
  raft::device_matrix_view<const float, int> local_dataset =
      load_local_dataset(rank);

  kmeans::params params;
  params.n_clusters = 1024;
  params.max_iter = 300;
  params.tol = 1e-4;

  auto centroids =
      raft::make_device_matrix<float, int>(
          handle, params.n_clusters, local_dataset.extent(1));
  auto labels =
      raft::make_device_vector<int, int>(handle, local_dataset.extent(0));

  float inertia;
  int n_iter;

  kmeans::fit(handle,
              params,
              local_dataset,
              std::nullopt,
              centroids.view(),
              raft::make_host_scalar_view(&inertia),
              raft::make_host_scalar_view(&n_iter));

  kmeans::predict(handle,
                  params,
                  local_dataset,
                  std::nullopt,
                  centroids.view(),
                  labels.view(),
                  true,
                  raft::make_host_scalar_view(&inertia));
}
```

</Tab>
</Tabs>

The NCCL communicator setup depends on the application runtime. The important NVIDIA cuVS requirement is that every participating rank uses a handle with the same communicator, world size, and rank metadata before calling the K-Means API.

### Assigning labels

Prediction assigns each row to the nearest learned centroid. Use it after fitting when you need a cluster label per row.

<Tabs>
<Tab title="C">

```c
#include <cuvs/cluster/kmeans.h>

cuvsResources_t res;
cuvsKMeansParams_t params;
DLManagedTensor *dataset;
DLManagedTensor *centroids;
DLManagedTensor *labels;
double inertia;

load_dataset(dataset);
load_centroids(centroids);
allocate_labels(labels);

cuvsResourcesCreate(&res);
cuvsKMeansParamsCreate(&params);

params->n_clusters = 1024;

cuvsKMeansPredict(res, params, dataset, NULL, centroids, labels, true, &inertia);

cuvsKMeansParamsDestroy(params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <optional>

using namespace cuvs::cluster;

raft::device_resources res;
raft::device_matrix_view<const float, int> dataset = load_dataset();
raft::device_matrix_view<const float, int> centroids = load_centroids();
auto labels = raft::make_device_vector<int, int>(res, dataset.extent(0));

kmeans::params params;
params.n_clusters = centroids.extent(0);

float inertia;

kmeans::predict(res,
                params,
                dataset,
                std::nullopt,
                centroids,
                labels.view(),
                true,
                raft::make_host_scalar_view(&inertia));
```

</Tab>
<Tab title="Python">

```python
from cuvs.cluster.kmeans import KMeansParams, fit, predict

params = KMeansParams(n_clusters=1024)
centroids, _, _ = fit(params, dataset)

labels, inertia = predict(params, dataset, centroids)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::cluster::kmeans::{self, Params};
use cuvs::{ManagedTensor, Resources, Result};

fn predict_kmeans(
    res: &Resources,
    params: &Params,
    dataset: &ManagedTensor,
    centroids: &ManagedTensor,
    n_samples: usize,
) -> Result<()> {
    let labels_host = ndarray::Array::<i32, _>::zeros((n_samples,));
    let mut labels = ManagedTensor::from(&labels_host).to_device(res)?;

    let _inertia =
        kmeans::predict(res, params, dataset, &None, centroids, &mut labels, true)?;
    Ok(())
}
```

</Tab>
</Tabs>

### Evaluating centroids

The cluster cost, also called inertia, is the sum of squared distances from each row to its closest centroid. Use it to compare different runs or to check whether additional iterations meaningfully improve the clustering.

<Tabs>
<Tab title="C">

```c
double cost;
cuvsKMeansClusterCost(res, dataset, centroids, &cost);
```

</Tab>
<Tab title="C++">

```cpp
float cost;
kmeans::cluster_cost(res,
                     dataset,
                     centroids,
                     raft::make_host_scalar_view(&cost));
```

</Tab>
<Tab title="Python">

```python
from cuvs.cluster.kmeans import cluster_cost

inertia = cluster_cost(dataset, centroids)
```

</Tab>
<Tab title="Rust">

```rust
let inertia = kmeans::cluster_cost(&res, &dataset, &centroids)?;
```

</Tab>
</Tabs>

## How K-Means works

K-Means alternates between two steps:

1. Assign each row to the closest centroid.
2. Move each centroid to the average of the rows assigned to it.

The algorithm repeats those steps until it reaches `max_iter` or the inertia changes by less than `tol`. The GPU is useful because the expensive part is repeatedly comparing many rows against many centroids.

## When to use K-Means

Use K-Means when you need compact representatives for a dataset, cluster labels for downstream analysis, coarse partitions for batching or indexing, or vector quantization codebooks.

K-Means works best when clusters are roughly spherical under the selected distance metric. If the data has complex shapes, very uneven cluster sizes, or strong outliers, consider using K-Means as a fast preprocessing step rather than treating the labels as final ground truth.

## Standard and balanced K-Means

Standard K-Means minimizes inertia. It can produce uneven cluster sizes if the data distribution is uneven.

Balanced K-Means encourages more even cluster sizes. It is useful when clusters will be used as partitions for later work and very large clusters would create load imbalance. In NVIDIA cuVS, balanced K-Means is exposed through `balanced_params` in C++ and through `hierarchical=True` in C and Python.

## Configuration parameters

### Fit parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / language default | Distance metric used to compare rows and centroids. Standard K-Means commonly uses squared L2 distance. |
| `n_clusters` | `8` | Number of clusters and output centroids. Larger values create finer groups but increase work and centroid memory. |
| `init` / `init_method` | `KMeansPlusPlus` | Initialization strategy. Use k-means++ for robust default seeding, random for faster but less stable seeding, or array initialization when providing centroids. |
| `max_iter` | `300` | Maximum number of training iterations for one run. |
| `tol` | `1e-4` | Relative inertia tolerance used for convergence. |
| `n_init` | `1` | Number of independent runs with different seeds. More runs can improve quality but multiply build time. |
| `oversampling_factor` | `2.0` | Oversampling factor used by k-means parallel initialization. |
| `batch_samples` | `32768` | Number of samples per tile for the nearest-centroid computation. Lower values reduce temporary memory. |
| `batch_centroids` | `0` | Number of centroids per tile. `0` means all centroids. Lower values reduce temporary memory. |
| `init_size` | `0` | Number of rows sampled for k-means++ initialization on host-data paths. `0` uses the default heuristic. |
| `streaming_batch_size` | `0` | Number of host rows streamed to the GPU per batch. `0` processes all host rows at once. |
| `hierarchical` | `false` | Enables hierarchical, balanced K-Means in C and Python. |
| `hierarchical_n_iters` | implementation default | Number of training iterations for hierarchical K-Means. |

## Tuning

Start with `n_clusters`. More clusters reduce average within-cluster distance, but they increase memory and the work per iteration. If clusters become too small or unstable, reduce `n_clusters` or increase the amount of data used for fitting.

Use `KMeansPlusPlus` initialization for most workloads. Increase `n_init` when quality matters more than fit time, especially when different random seeds produce noticeably different inertia.

Tune `max_iter` and `tol` together. If `n_iter` often reaches `max_iter`, increase `max_iter` or relax `tol`. If inertia stops improving early, lowering `max_iter` can reduce runtime.

Use `batch_samples` and `batch_centroids` to control device memory for device-resident data. Smaller tiles reduce temporary memory but add more tiled work.

Use `streaming_batch_size` when fitting host-resident datasets that do not fit on the GPU. Smaller batches reduce GPU memory pressure, while larger batches reduce transfer and launch overhead.

Use balanced K-Means when cluster size matters. This is often useful when clusters are later used as work partitions, batches, or coarse groups for another algorithm.

## Memory footprint

K-Means memory is dominated by the input data, the centroid matrix, optional labels or weights, and temporary tiles used to compare samples with centroids. The exact scratch space depends on the selected data type and implementation path, but the estimates below are useful for planning.

Variables:

- `N`: Number of rows, or samples.
- `D`: Number of features per row.
- `K`: Number of clusters, or centroids.
- `S`: Number of samples processed in one GPU batch.
- `T_s`: Sample tile size, controlled by `batch_samples`.
- `T_c`: Centroid tile size, controlled by `batch_centroids`; if `batch_centroids = 0`, use `K`.
- `B_x`: Bytes per input element.
- `B_c`: Bytes per centroid element.
- `B_l`: Bytes per output label.

### Scratch and maximum rows

The `scratch` term covers temporary buffers that are not part of the persistent inputs or outputs: reduction buffers, assignment buffers, allocator padding, CUDA library workspaces, and memory held by the active memory resource. For a first capacity estimate, reserve a scratch headroom factor `H`. Use `H = 0.20` for K-Means fit and `H = 0.10` for prediction. If you can measure a representative run, use:

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


To estimate the largest usable row count, rewrite the selected peak formula as:

$$
\text{peak\_without\_scratch}(N)
  = N \cdot B_{\text{per\_row}} + B_{\text{fixed}}
$$

and solve:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_row}}}
  \right\rfloor
$$

For host streaming fit, solve for `S` first. `S` controls the active GPU batch size even when the full dataset has more than `S` rows.

### Device-resident fit

When the dataset is already on the GPU, the persistent data and centroids use:

$$
\begin{aligned}
\text{dataset\_size} &= N \cdot D \cdot B_x \\
\text{centroids\_size} &= K \cdot D \cdot B_c
\end{aligned}
$$

The nearest-centroid computation is tiled. A useful estimate for the main distance tile is:

$$
\text{distance\_tile\_size}
  \approx \min(N, T_s) \cdot \min(K, T_c) \cdot B_c
$$

The fit peak is approximately:

$$
\begin{aligned}
\text{fit\_peak}
  \approx&\ \text{dataset\_size}
   + \text{centroids\_size} \\
  &+ \text{distance\_tile\_size}
   + \text{scratch}
\end{aligned}
$$

### Host streaming fit

For host-resident data, NVIDIA cuVS can stream rows to the GPU in batches. If `streaming_batch_size = 0`, then `S = N`; otherwise `S = streaming_batch_size`.

$$
\begin{aligned}
\text{streaming\_batch\_size}
  &= S \cdot D \cdot B_x \\
\text{fit\_peak}
  \approx&\ \text{streaming\_batch\_size}
   + \text{centroids\_size} \\
  &+ \min(S, T_s) \cdot \min(K, T_c) \cdot B_c
   + \text{scratch}
\end{aligned}
$$

Use a smaller `streaming_batch_size` when the host-data fit path runs out of GPU memory. Use a larger value when GPU memory is available and transfer overhead dominates.

### Prediction

Prediction needs the input rows, centroids, output labels, and a distance tile:

$$
\begin{aligned}
\text{labels\_size} &= N \cdot B_l \\
\text{predict\_peak}
  \approx&\ N \cdot D \cdot B_x
   + K \cdot D \cdot B_c \\
  &+ \text{labels\_size}
   + \text{distance\_tile\_size}
   + \text{scratch}
\end{aligned}
$$

For large `K`, reduce `batch_centroids`. For large `N`, reduce `batch_samples` or use the host streaming fit path when fitting from host data.
