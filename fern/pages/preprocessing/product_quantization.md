# Product Quantization

Product Quantization, or PQ, is a GPU-accelerated preprocessing algorithm for compressing vectors. It learns small codebooks from a training dataset, then represents each vector with compact integer codes instead of storing every original floating-point value.

Use PQ when memory footprint or bandwidth is the bottleneck and approximate reconstruction is acceptable. PQ is not an ANN index by itself. It is a compression step that can be used before storage, approximate scoring, vector quantization workflows, or index construction.

## Example API Usage

[C API](/api-reference/c-api-preprocessing-quantize-pq) | [C++ API](/api-reference/cpp-api-preprocessing-quantize-pq) | [Python API](/api-reference/python-api-preprocessing-quantize-pq)

### Building a quantizer

Building trains the PQ codebooks from a representative dataset. The trained quantizer can then transform compatible datasets into compact PQ codes.

<Tabs>
<Tab title="C">

```c
#include <cuvs/preprocessing/quantize/pq.h>
#include <cuvs/core/c_api.h>

cuvsResources_t res;
cuvsProductQuantizerParams_t params;
cuvsProductQuantizer_t quantizer;
DLManagedTensor *dataset;

load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsProductQuantizerParamsCreate(&params);
cuvsProductQuantizerCreate(&quantizer);

params->pq_bits = 8;
params->pq_dim = 16;
params->use_subspaces = true;
params->use_vq = false;

cuvsProductQuantizerBuild(res, params, dataset, quantizer);

cuvsProductQuantizerDestroy(quantizer);
cuvsProductQuantizerParamsDestroy(params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace pq = cuvs::preprocessing::quantize::pq;

raft::device_resources res;
raft::device_matrix_view<const float, int64_t> dataset = load_dataset();

pq::params params;
params.pq_bits = 8;
params.pq_dim = 16;
params.use_subspaces = true;
params.use_vq = false;

auto quantizer = pq::build(res, params, dataset);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp

from cuvs.preprocessing.quantize import pq

dataset = cp.asarray(load_dataset(), dtype=cp.float32)
params = pq.QuantizerParams(pq_bits=8, pq_dim=16, use_subspaces=True)

quantizer = pq.build(params, dataset)
```

</Tab>
</Tabs>

### Transforming data

Transforming replaces each vector with its PQ code. If vector quantization is enabled with `use_vq=True`, the transform also returns one VQ label per row.

<Tabs>
<Tab title="C">

```c
uint32_t encoded_dim;
DLManagedTensor *codes;

cuvsProductQuantizerGetEncodedDim(quantizer, &encoded_dim);
allocate_codes(codes, n_rows, encoded_dim);

cuvsProductQuantizerTransform(res, quantizer, dataset, codes, NULL);
```

</Tab>
<Tab title="C++">

```cpp
auto encoded_dim = pq::get_quantized_dim(params);
auto codes = raft::make_device_matrix<uint8_t, int64_t>(
    res, dataset.extent(0), encoded_dim);

pq::transform(res, quantizer, dataset, codes.view());
```

</Tab>
<Tab title="Python">

```python
codes, vq_labels = pq.transform(quantizer, dataset)
```

</Tab>
</Tabs>

### Reconstructing vectors

Inverse transform reconstructs approximate floating-point vectors from PQ codes. Reconstruction is lossy because each code stores the nearest learned codebook entry, not the original subvector.

<Tabs>
<Tab title="C">

```c
DLManagedTensor *reconstructed;

allocate_reconstructed(reconstructed, n_rows, n_features);
cuvsProductQuantizerInverseTransform(
    res, quantizer, codes, reconstructed, NULL);
```

</Tab>
<Tab title="C++">

```cpp
auto reconstructed =
    raft::make_device_matrix<float, int64_t>(res, dataset.extent(0), dataset.extent(1));

pq::inverse_transform(res, quantizer, codes.view(), reconstructed.view());
```

</Tab>
<Tab title="Python">

```python
reconstructed = pq.inverse_transform(quantizer, codes, vq_labels=vq_labels)
```

</Tab>
</Tabs>

## How Product Quantization works

PQ divides each vector into smaller subvectors. For each subvector position, it trains a codebook with `2^pq_bits` representative values. Transforming a vector stores the ID of the nearest codebook entry for each subvector.

This turns a high-dimensional floating-point vector into a much shorter code. For example, if `pq_bits = 8` and `pq_dim = 16`, each vector is represented by 16 one-byte codes.

PQ can also be combined with vector quantization. With `use_vq=True`, NVIDIA cuVS first assigns each vector to a coarse VQ centroid, then trains PQ on the residuals. This can improve reconstruction quality when the dataset has strong global structure, at the cost of extra labels and VQ codebook memory.

## When to use Product Quantization

Use PQ when vectors are too large to store or move efficiently in full precision, when approximate reconstruction is acceptable, or when a downstream workflow can use compact codes instead of original vectors.

PQ is especially useful for large vector search systems because it reduces memory traffic. It can also be useful for compression before storage, coarse candidate scoring, or as a building block inside ANN indexes such as IVF-PQ.

Avoid PQ when exact vector values are required. PQ is lossy, so increasing compression usually reduces reconstruction quality and can reduce recall in downstream search workflows.

## Configuration parameters

### Build parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `pq_bits` | `8` | Number of bits per PQ code. Higher values improve reconstruction quality but increase code size and codebook size. Valid standalone PQ values are `[4, 16]`. |
| `pq_dim` | `0` | Number of PQ code dimensions, or subquantizers. `0` selects a heuristic. The input dimension currently needs to be compatible with `pq_dim`. |
| `use_subspaces` | `true` | When true, trains a separate codebook for each subspace. When false, uses one shared codebook. |
| `use_vq` | `false` | Enables a coarse vector quantizer before PQ. PQ is then trained on residuals. |
| `vq_n_centers` | `0` | Number of VQ centroids. `0` selects a heuristic. `1` effectively disables VQ. |
| `kmeans_n_iters` | `25` | Number of k-means iterations used during VQ and PQ codebook training. |
| `pq_kmeans_type` | `kmeans_balanced` | K-Means variant used to train PQ codebooks. Balanced K-Means is the default. |
| `max_train_points_per_pq_code` | `256` | Maximum number of training rows per PQ code. Larger values can improve codebooks but increase build time. |
| `max_train_points_per_vq_cluster` | `1024` | Maximum number of training rows per VQ cluster when VQ is enabled. |

## Tuning

Start with `pq_bits = 8`. Lower values reduce memory but make each code less precise. Higher values can improve reconstruction quality but increase code size and codebook memory.

Choose `pq_dim` based on the desired code size and the input dimension. More PQ dimensions usually improve reconstruction quality because each code represents a smaller subvector, but the encoded row becomes longer.

Keep `use_subspaces=True` for most workloads. Separate subspace codebooks usually improve quality because each part of the vector gets its own codebook.

Enable `use_vq` when a single global PQ codebook is not accurate enough and the data has meaningful coarse groups. VQ adds a coarse label and VQ codebook, so it improves quality at the cost of more metadata and training work.

Increase `max_train_points_per_pq_code` or `kmeans_n_iters` when reconstruction quality is poor and build time is acceptable. Decrease them when build time is the main constraint.

Use `kmeans_balanced` for PQ codebook training unless there is a specific reason to prefer regular K-Means.

## Memory footprint

PQ memory is dominated by the encoded output, PQ codebooks, optional VQ labels, and temporary training buffers. The estimates below use bytes and are meant for planning.

Variables:

- `N`: Number of vectors.
- `D`: Number of input features.
- `P`: PQ dimension, or number of PQ code dimensions.
- `b`: Bits per PQ code, or `pq_bits`.
- `C`: Number of entries in each PQ codebook, equal to `2^b`.
- `S`: Subvector width, approximately `D / P`.
- `L`: Number of VQ centers, or `vq_n_centers`.
- `B_x`: Bytes per input element.
- `B_c`: Bytes per codebook element.
- `B_l`: Bytes per VQ label.

### Scratch and maximum rows

The `scratch` term covers K-Means workspace for codebook training, residual buffers, transform buffers, allocator padding, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.20` for build and `H = 0.10` for transform-only planning. If you can measure a representative run, use:

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


Most PQ storage terms are linear in `N`. Rewrite the selected peak as:

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

Training buffers use `min(...)` terms. If the cap is active, treat the capped training term as fixed; otherwise include it in `B_per_row`.

### Encoded vectors

Each row stores `P` codes with `b` bits per code:

$$
\begin{aligned}
\text{encoded\_dim}
  &= \left\lceil \frac{P \cdot b}{8} \right\rceil \\
\text{codes\_size}
  &= N \cdot \text{encoded\_dim}
\end{aligned}
$$

This is the main persistent output of PQ.

### Codebooks

With separate subspace codebooks, the PQ codebook size is:

$$
\text{pq\_codebook\_size}
  \approx P \cdot C \cdot S \cdot B_c
$$

When `use_subspaces=False`, a shared codebook is used:

$$
\text{shared\_codebook\_size}
  \approx C \cdot S \cdot B_c
$$

If VQ is enabled, add the coarse VQ codebook:

$$
\text{vq\_codebook\_size}
  \approx L \cdot D \cdot B_c
$$

### VQ labels

When `use_vq=True`, transform also stores one VQ label per row:

$$
\text{vq\_labels\_size}
  = N \cdot B_l
$$

### Training peak

PQ training samples up to `C * max_train_points_per_pq_code` rows for each PQ codebook. A rough training buffer estimate per codebook is:

$$
\text{pq\_training\_buffer}
  \approx
  \min(N,\ C \cdot \text{max\_train\_points\_per\_pq\_code})
  \cdot S \cdot B_x
$$

If VQ is enabled, VQ training can additionally use:

$$
\text{vq\_training\_buffer}
  \approx
  \min(N,\ L \cdot \text{max\_train\_points\_per\_vq\_cluster})
  \cdot D \cdot B_x
$$

The overall build peak is approximately:

$$
\begin{aligned}
\text{build\_peak}
  \approx&\ N \cdot D \cdot B_x
   + \text{pq\_codebook\_size} \\
  &+ \text{pq\_training\_buffer}
   + \text{optional\_vq\_buffers}
   + \text{scratch}
\end{aligned}
$$

For large datasets, reduce `max_train_points_per_pq_code`, reduce `max_train_points_per_vq_cluster`, or disable VQ to reduce training memory and build time.
