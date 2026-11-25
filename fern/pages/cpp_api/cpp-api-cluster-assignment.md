---
slug: api-reference/cpp-api-cluster-assignment
---

# Assignment

_Source header: `cuvs/cluster/assignment.hpp`_

## Nearest-centroid assignment backends

<a id="cluster-ivf-assignment-params"></a>
### cluster::ivf_assignment_params

Parameters of the IVF-based (approximate) nearest-centroid assignment.

An approximate (ANN) assignment that runs an IVF-Flat style search over the centroids with k = 1. The centroids are grouped into `n_lists` lists (about sqrt(n_centroids)) by balanced k-means. Each vector scores the list centers, probes its `n_probes` best lists and takes the best centroid among them. Unlike a per-query IVF-Flat search, the work is organized by list: the vectors are bucketed by the lists they probe, and each list is scored against its centroids with dense GEMMs on tensor cores (TF32), or with one fused kernel. The per-vector cost is about (n_lists + n_probes * n_centroids / n_lists) * dim multiply-adds, versus n_centroids * dim for an exhaustive scan, so it pays off when there are many centroids. The result is approximate: a vector whose nearest centroid lies in a list it does not probe gets the best centroid among the lists it does probe. Probing every list (`n_probes >= n_lists`) is exhaustive. The data must have a dimension divisible by 4.

Measured on an H100 with 64K to 128K vectors per call, the default number of lists and 16 probes, on datasets of dimension 96 to 2048, the assignment ran at about 0.3x to 0.8x the speed of the exhaustive scan at 4K centroids, 0.6x to 2.9x at 16K, 1.1x to 4.7x at 32K, 3.4x to 7.8x at 65K and 8x to 48x at 262K to 1M, with at least 99% of the labels equal to the exhaustive scan's up to 131K centroids (98.3% at 1M). Fewer centroids favor the exhaustive scan. The gain depends on the data and the GPU, so benchmark on yours.

```cpp
struct ivf_assignment_params {
  uint32_t n_lists;
  uint32_t kmeans_n_iters;
  uint32_t n_probes;
  const ivf_assignment_index* index;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_lists` | `uint32_t` | Number of lists. 0 means round(sqrt(n_centroids)). Only used when building an index. |
| `kmeans_n_iters` | `uint32_t` | Balanced k-means iterations used to group the centroids. Only used when building. |
| `n_probes` | `uint32_t` | Lists probed per vector (clamped to the index's n_lists). 0 selects 16, a rule of thumb rather than an accuracy guarantee. Used on every call. |
| `index` | [`const ivf_assignment_index*`](/api-reference/cpp-api-cluster-assignment#cluster-ivf-assignment-index) | Optional prebuilt index, not owned (must outlive the call). If null, an index is built from the centroids on every call and discarded; build once with `build_ivf_assignment_index` and pass it here to reuse it across calls. When set, `n_lists` and `kmeans_n_iters` are ignored and the number of centroids, the dimension and the metric must match the call (else it throws). The index is only read, so it may be shared by concurrent calls. |

<a id="assignment-params"></a>
### assignment_params

Selects how vectors are assigned to their nearest centroid.

```cpp
using assignment_params = std::variant<ivf_assignment_params>;
```

<a id="cluster-ivf-assignment-index"></a>
### cluster::ivf_assignment_index

Index of the IVF assignment, built from a fixed set of centroids.

Opaque, move-only, not serializable. It holds a regrouped TF32 copy of the centroids on the device.

```cpp
class ivf_assignment_index;
```

<a id="cluster-ivf-assignment-index-n-centroids"></a>
### cluster::ivf_assignment_index::n_centroids

Number of centroids the index was built from.

```cpp
uint32_t n_centroids() const noexcept;
```

**Returns**

`uint32_t`

<a id="cluster-ivf-assignment-index-dim"></a>
### cluster::ivf_assignment_index::dim

Dimension of the centroids.

```cpp
uint32_t dim() const noexcept;
```

**Returns**

`uint32_t`

<a id="cluster-ivf-assignment-index-n-lists"></a>
### cluster::ivf_assignment_index::n_lists

Number of lists.

```cpp
uint32_t n_lists() const noexcept;
```

**Returns**

`uint32_t`

<a id="cluster-ivf-assignment-index-metric"></a>
### cluster::ivf_assignment_index::metric

Metric the index was built for.

```cpp
cuvs::distance::DistanceType metric() const noexcept;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="cluster-build-ivf-assignment-index"></a>
### cluster::build_ivf_assignment_index

Build an IVF assignment index from centroids.

```cpp
auto build_ivf_assignment_index(
raft::resources const& res,
const ivf_assignment_params& params,
raft::device_matrix_view<const float, int64_t, raft::row_major> centroids,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
-> ivf_assignment_index;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const ivf_assignment_params&`](/api-reference/cpp-api-cluster-assignment#cluster-ivf-assignment-params) | `n_lists` and `kmeans_n_iters` are used; the rest is ignored |
| `centroids` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | [n_centroids, dim] row-major, dim divisible by 4 |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | L2Expanded, L2SqrtExpanded, CosineExpanded or InnerProduct<br />Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

[`ivf_assignment_index`](/api-reference/cpp-api-cluster-assignment#cluster-ivf-assignment-index)
