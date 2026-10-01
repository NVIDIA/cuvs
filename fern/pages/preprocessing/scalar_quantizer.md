# Scalar Quantizer

Scalar quantization compresses floating-point vectors by mapping each value into a smaller integer type. In NVIDIA cuVS, the scalar quantizer learns a global floating-point range and then linearly maps values in that range to `int8`.

Use scalar quantization when full-precision vectors are too expensive to store or move, but you still want a simple element-wise approximation that can be reconstructed later. It is often useful before storage, approximate scoring, or ANN workflows that can tolerate quantization error.

## Example API Usage

[C API](/api-reference/c-api-preprocessing-quantize-scalar) | [C++ API](/api-reference/cpp-api-preprocessing-quantize-scalar) | [Python API](/api-reference/python-api-preprocessing-quantize-scalar)

### Training a quantizer

Training finds the value range used for the linear mapping. The `quantile` parameter can ignore extreme high and low values so that a small number of outliers do not stretch the range for the entire dataset.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/scalar.h>

cuvsResources_t res;
cuvsScalarQuantizerParams_t params;
cuvsScalarQuantizer_t quantizer;
DLManagedTensor *dataset;

load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsScalarQuantizerParamsCreate(&params);
cuvsScalarQuantizerCreate(&quantizer);

params->quantile = 0.99f;

cuvsScalarQuantizerTrain(res, params, dataset, quantizer);

cuvsScalarQuantizerDestroy(quantizer);
cuvsScalarQuantizerParamsDestroy(params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/preprocessing/quantize/scalar.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace scalar = cuvs::preprocessing::quantize::scalar;

raft::device_resources res;
raft::device_matrix_view<const float, int64_t> dataset = load_dataset();

scalar::params params;
params.quantile = 0.99f;

auto quantizer = scalar::train(res, params, dataset);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp

from cuvs.preprocessing.quantize import scalar

dataset = cp.asarray(load_dataset(), dtype=cp.float32)
params = scalar.QuantizerParams(quantile=0.99)

quantizer = scalar.train(params, dataset)
```

</Tab>
</Tabs>

### Transforming data

Transforming replaces each floating-point value with an `int8` code. Inverse transform maps those codes back into approximate floating-point values.

<Tabs>
<Tab title="C">

```c
DLManagedTensor *codes;
DLManagedTensor *reconstructed;

allocate_int8_matrix(codes, n_rows, n_features);
allocate_float_matrix(reconstructed, n_rows, n_features);

cuvsScalarQuantizerTransform(res, quantizer, dataset, codes);
cuvsScalarQuantizerInverseTransform(res, quantizer, codes, reconstructed);
```

</Tab>
<Tab title="C++">

```cpp
auto codes = raft::make_device_matrix<int8_t, int64_t>(
    res, dataset.extent(0), dataset.extent(1));
auto reconstructed = raft::make_device_matrix<float, int64_t>(
    res, dataset.extent(0), dataset.extent(1));

scalar::transform(res, quantizer, dataset, codes.view());
scalar::inverse_transform(res, quantizer, codes.view(), reconstructed.view());
```

</Tab>
<Tab title="Python">

```python
codes = scalar.transform(quantizer, dataset)
reconstructed = scalar.inverse_transform(quantizer, codes)
```

</Tab>
</Tabs>

## How Scalar Quantization works

The quantizer stores a minimum and maximum value. Each input value is clipped to that range and mapped into the available `int8` codes. Inverse transform applies the reverse mapping, so reconstructed values are approximate rather than exact.

The range is learned from the training data. With `quantile = 1.0`, the full observed range is used. With a smaller quantile, NVIDIA cuVS ignores a small fraction of values at the high and low ends before choosing the range.

## When to use Scalar Quantization

Use scalar quantization when memory bandwidth or storage size is more important than exact reconstruction. It is a good fit when every dimension has roughly comparable scale, or when the downstream algorithm is robust to small per-value errors.

Avoid scalar quantization when exact values are required, when very small numeric differences matter, or when a few dimensions have very different ranges and need more specialized handling.

## Configuration parameters

### Train parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `quantile` | `0.99` | Fraction of the distribution used to choose the quantization range. Lower values reduce the influence of outliers but can clip more values. Must be in `(0, 1]`. |

## Tuning

Start with `quantile = 0.99`. Increase it toward `1.0` when clipping hurts reconstruction quality. Decrease it when a few outliers stretch the range and make most values less precise.

Check downstream recall or reconstruction error rather than only looking at compression ratio. Scalar quantization always stores one byte per value, so the main tradeoff is how well the learned range matches the data.

## Memory footprint

Scalar quantization memory is dominated by the input matrix and the `int8` output matrix.

Variables:

- `N`: Number of vectors.
- `D`: Number of features per vector.
- `B_x`: Bytes per input element.
- `B_q`: Bytes per quantized output element. For `int8`, `B_q = 1`.
- `B_c`: Bytes per stored quantizer value.

### Scratch and maximum rows

The `scratch` term covers temporary transform buffers, allocator padding, CUDA library workspaces, and memory held by the active memory resource. Scalar quantization is mostly a streaming transform, so use `H = 0.10` as a first scratch headroom estimate. If you can measure a representative run, use:

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


The transform formulas are linear in `N`. Rewrite the selected peak as:

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

### Persistent data

The quantized dataset stores one byte per input value:

$$
\text{codes\_size}
  = N \cdot D \cdot B_q
$$

The scalar quantizer stores the learned range:

$$
\text{quantizer\_size}
  \approx 2 \cdot B_c
$$

### Transform peak

The transform peak is approximately:

$$
\begin{aligned}
\text{transform\_peak}
  \approx&\ N \cdot D \cdot B_x
   + \text{codes\_size} \\
  &+ \text{quantizer\_size}
   + \text{scratch}
\end{aligned}
$$

Inverse transform replaces `codes_size` with the reconstructed floating-point output:

$$
\text{reconstructed\_size}
  = N \cdot D \cdot B_x
$$
