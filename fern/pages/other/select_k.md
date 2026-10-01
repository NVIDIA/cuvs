# K-selection

K-selection selects the `k` smallest or largest values from each row of an input matrix. It also returns payload indices for the selected values, so it can turn a score or distance matrix into top-k results.

Use K-selection when you already have per-row scores, distances, or candidate values and need to keep only the best `k` entries per row.

## Example API Usage

[C++ API](/api-reference/cpp-api-selection-select-k)

### Selecting smallest values

For distance matrices, nearest neighbors are usually the smallest values in each row.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/selection/select_k.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

raft::device_resources res;
raft::device_matrix_view<const float, int64_t, raft::row_major> distances =
    load_distance_matrix();
int64_t k = 10;

auto out_values = raft::make_device_matrix<float, int64_t>(
    res, distances.extent(0), k);
auto out_indices = raft::make_device_matrix<int64_t, int64_t>(
    res, distances.extent(0), k);

cuvs::selection::select_k(res,
                          distances,
                          std::nullopt,
                          out_values.view(),
                          out_indices.view(),
                          true,
                          true);
```

</Tab>
</Tabs>

### Selecting largest values

For similarity scores, best matches are often the largest values.

<Tabs>
<Tab title="C++">

```cpp
auto out_scores = raft::make_device_matrix<float, int64_t>(
    res, scores.extent(0), k);
auto out_ids = raft::make_device_matrix<uint32_t, int64_t>(
    res, scores.extent(0), k);

cuvs::selection::select_k(res,
                          scores,
                          candidate_ids,
                          out_scores.view(),
                          out_ids.view(),
                          false,
                          true);
```

</Tab>
</Tabs>

### Variable row lengths

Use `len_i` when each row has a different valid length, such as padded candidate lists.

<Tabs>
<Tab title="C++">

```cpp
std::optional<raft::device_vector_view<const int64_t, int64_t>> row_lengths =
    valid_lengths.view();

cuvs::selection::select_k(res,
                          candidates,
                          candidate_ids,
                          out_values.view(),
                          out_indices.view(),
                          true,
                          false,
                          cuvs::selection::SelectAlgo::kAuto,
                          row_lengths);
```

</Tab>
</Tabs>

## How K-selection works

K-selection processes each row independently. For each row, it keeps the best `k` values according to `select_min`. If `select_min=true`, lower values are better. If `select_min=false`, higher values are better.

The output values and output indices have shape `batch_size x k`. When `in_idx` is omitted, the selected payload is the column index from the input row. When `in_idx` is provided, the output payload comes from the corresponding entry in `in_idx`.

## When to use K-selection

Use K-selection after computing scores or distances when the full matrix is too large or noisy to keep. Common examples include exact nearest-neighbor search, reranking candidate lists, selecting top scoring documents, pruning graph candidates, and keeping top-k model outputs.

K-selection is also useful inside multi-stage workflows. A first stage can produce many candidates, then K-selection keeps a smaller set for refinement, reranking, or downstream processing.

Avoid using K-selection as a substitute for indexing when the expensive part is computing all distances. K-selection reduces output size, but it does not avoid the cost of generating the input values.

## Tuning

Start with `SelectAlgo::kAuto`. The automatic choice is the safest default across different `k`, row lengths, and data types.

Set `sorted=false` when result ordering is not needed. Sorting selected values adds work, and many downstream stages only need the selected set.

Use the smallest `k` that satisfies the downstream workflow. Selection cost and output memory both grow with `k`.

Use `len_i` instead of sentinel values when rows have different valid lengths. This avoids selecting padded values and can reduce unnecessary work.

## Memory footprint

K-selection memory is dominated by the input matrix and the selected output matrices.

Variables:

- `B`: Batch size, or number of rows.
- `L`: Number of input values per row.
- `K`: Number of selected values per row.
- `B_v`: Bytes per value.
- `B_i`: Bytes per payload index.

### Scratch and maximum rows

The `scratch` term covers temporary selection buffers, optional sorting workspace, allocator padding, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.10` for unsorted selection and `H = 0.15` when `sorted=true`. If you can measure a representative run, use:

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


K-selection scales with batch rows `B`, not database vectors directly. For fixed row length `L` and selected count `K`, rewrite the peak as:

$$
\text{peak\_without\_scratch}(B)
  = B \cdot B_{\text{per\_row}} + B_{\text{fixed}}
$$

and solve:

$$
B_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_row}}}
  \right\rfloor
$$

For nearest-neighbor workflows, this `B_max` is the maximum number of query rows that can be selected from one materialized candidate matrix.

### Inputs and outputs

The input values use:

$$
\text{input\_values\_size}
  = B \cdot L \cdot B_v
$$

If `in_idx` is provided, input payloads use:

$$
\text{input\_payload\_size}
  = B \cdot L \cdot B_i
$$

The output matrices use:

$$
\text{output\_size}
  = B \cdot K \cdot (B_v + B_i)
$$

The peak memory is approximately:

$$
\begin{aligned}
\text{select\_peak}
  \approx&\ \text{input\_values\_size}
   + \text{input\_payload\_size} \\
  &+ \text{output\_size}
   + \text{scratch}
\end{aligned}
$$

For exact nearest-neighbor workflows, avoid materializing both a large distance matrix and large selected outputs at the same time when tiled distance computation can select and merge candidates incrementally.
