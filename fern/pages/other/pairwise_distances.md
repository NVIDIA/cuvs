# Pairwise Distances

The pairwise distances API computes a dense distance matrix between two sets of vectors. Given `X` with `N` rows and `Y` with `M` rows, it writes an `N x M` matrix where each entry is the distance between one row of `X` and one row of `Y`.

Use pairwise distances when the distance matrix itself is the output, or when a downstream algorithm needs all cross-distances before selecting, clustering, scoring, or evaluating results.

## Example API Usage

[C API](/api-reference/c-api-distance-pairwise-distance) | [C++ API](/api-reference/cpp-api-distance-distance) | [Python API](/api-reference/python-api-distance)

### Computing dense distances

The dense path supports contiguous row-major or column-major matrices. The two input matrices must have the same number of columns.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/distance/pairwise_distance.h>

cuvsResources_t res;
DLManagedTensor *x;
DLManagedTensor *y;
DLManagedTensor *distances;

load_matrix(x);
load_matrix(y);
allocate_distance_matrix(distances);

cuvsResourcesCreate(&res);

cuvsPairwiseDistance(res,
                     x,
                     y,
                     distances,
                     L2SqrtExpanded,
                     2.0f);

cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

raft::device_resources res;
raft::device_matrix_view<const float, int64_t, raft::layout_c_contiguous> x =
    load_matrix_x();
raft::device_matrix_view<const float, int64_t, raft::layout_c_contiguous> y =
    load_matrix_y();

auto distances = raft::make_device_matrix<float, int64_t>(
    res, x.extent(0), y.extent(0));

cuvs::distance::pairwise_distance(
    res,
    x,
    y,
    distances.view(),
    cuvs::distance::DistanceType::L2SqrtExpanded);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp

from cuvs.distance import pairwise_distance

x = cp.asarray(load_matrix_x(), dtype=cp.float32)
y = cp.asarray(load_matrix_y(), dtype=cp.float32)

distances = pairwise_distance(x, y, metric="euclidean")
```

</Tab>
</Tabs>

### Computing sparse distances

The C++ API also supports CSR inputs for sparse pairwise distances.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

raft::device_resources res;
auto x = load_csr_matrix_x<float>();
auto y = load_csr_matrix_y<float>();

auto distances = raft::make_device_matrix<float, int>(
    res, x.structure_view().get_n_rows(), y.structure_view().get_n_rows());

cuvs::distance::pairwise_distance(
    res,
    x.view(),
    y.view(),
    distances.view(),
    cuvs::distance::DistanceType::L2Expanded);
```

</Tab>
</Tabs>

### Selecting nearest neighbors

Pairwise distance often feeds a selection step. Compute the full matrix when it fits in memory, then use [K-selection](/user-guide/api-guides/other-ap-is/k-selection) to keep the best values from each query row.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/selection/select_k.hpp>

auto neighbors = raft::make_device_matrix<int64_t, int64_t>(
    res, distances.extent(0), k);
auto neighbor_distances = raft::make_device_matrix<float, int64_t>(
    res, distances.extent(0), k);

cuvs::selection::select_k(res,
                          distances.view(),
                          std::nullopt,
                          neighbor_distances.view(),
                          neighbors.view(),
                          true,
                          true);
```

</Tab>
</Tabs>

## How Pairwise Distances works

Pairwise distance compares every row of `X` with every row of `Y`. If `X` has `N` rows and `Y` has `M` rows, the output has `N * M` distances.

For nearest-neighbor workflows, pairwise distance is the exact brute-force distance stage. It does not build an index, so there is no recall tradeoff. The cost is that runtime and output memory grow with the product of the two input row counts.

## When to use Pairwise Distances

Use pairwise distances when exact all-pairs or cross-pairs distances are required, when the distance matrix feeds another algorithm, or when you need a correctness baseline for approximate search.

It is also useful for batched scoring, evaluation, clustering inputs, small to medium exact search problems, and tests that need deterministic ground-truth distances.

Avoid materializing a full pairwise distance matrix when `N * M` is too large for memory. For nearest-neighbor search at larger scale, use an index or compute distances in batches and select candidates incrementally.

## Tuning

Start by checking output size. The distance matrix has `N * M` elements, so memory can grow faster than the input matrices.

Use `float32` unless the workload specifically needs `float64`. Half-precision inputs can reduce input bandwidth, but dense half inputs produce `float32` distances in C++ and Python.

Use squared L2 distance when the square root is not needed. For ranking nearest neighbors, squared L2 and L2 produce the same ordering, but squared L2 avoids the square root.

Batch the computation when `N * M` does not fit comfortably in memory. For nearest-neighbor workflows, combine each batch with selection so you do not need to keep the full distance matrix.

## Memory footprint

Pairwise distance memory is dominated by the output matrix.

Variables:

- `N`: Number of rows in `X`.
- `M`: Number of rows in `Y`.
- `D`: Number of features per row.
- `B_x`: Bytes per input element.
- `B_d`: Bytes per output distance.

### Scratch and maximum rows

The `scratch` term covers tiled distance-computation buffers, allocator padding, CUDA library workspaces, and memory held by the active memory resource. Pairwise distance is usually dominated by the output matrix, so use `H = 0.10` for a first estimate. If you can measure a representative run, use:

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


If `M` is fixed, solve for the maximum number of rows in `X`:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - M \cdot D \cdot B_x - B_{\text{fixed}}}
         {D \cdot B_x + M \cdot B_d}
  \right\rfloor
$$

If this is an all-pairs calculation where `M = N`, solve the quadratic term from the output matrix:

$$
N_{\max}
  \approx
  \left\lfloor
    \sqrt{\frac{M_{\text{usable}} - B_{\text{fixed}}}{B_d}}
  \right\rfloor
$$

This approximation is conservative only when `N * M * B_d` dominates the input matrices. For smaller problems, solve the full formula including input sizes.

### Dense inputs

The input matrices use:

$$
\text{input\_size}
  = (N \cdot D + M \cdot D) \cdot B_x
$$

The output matrix uses:

$$
\text{distance\_matrix\_size}
  = N \cdot M \cdot B_d
$$

The peak memory is approximately:

$$
\begin{aligned}
\text{pairwise\_peak}
  \approx&\ \text{input\_size}
   + \text{distance\_matrix\_size} \\
  &+ \text{scratch}
\end{aligned}
$$

When the distance matrix is too large, split `X` or `Y` into batches and process one output tile at a time.
