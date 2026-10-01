# ScaNN

ScaNN is an experimental NVIDIA cuVS index builder for the open-source ScaNN format. It combines partitioning, residual product quantization, SOAR spilling, and optional bfloat16 reordering data. Think of it as a pipeline that first sorts vectors into buckets, then stores compact shortcuts for the vectors in each bucket, and finally writes those pieces so OSS ScaNN can search them.

The NVIDIA cuVS SCaNN API currently builds and serializes indexes from C++. It does not expose an NVIDIA cuVS search API.

## Example API Usage

[C++ API](/api-reference/cpp-api-neighbors-scann)

### Building an index

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/scann.hpp>

using namespace cuvs::neighbors::experimental;

raft::device_resources res;
auto dataset = load_device_dataset();

scann::index_params index_params;
index_params.n_leaves = 1000;
index_params.kmeans_n_rows_train = 100000;
index_params.kmeans_n_iters = 24;
index_params.partitioning_eta = 1.0f;
index_params.soar_lambda = 1.0f;
index_params.pq_dim = 8;
index_params.pq_bits = 8;
index_params.pq_n_rows_train = 100000;
index_params.pq_train_iters = 10;

auto index = scann::build(res, index_params, dataset);
```

</Tab>
</Tabs>

The build API also accepts host row-major `float` data. C, Python, Java, Rust, and Go do not currently expose SCaNN bindings.

### Serializing an index

<Tabs>
<Tab title="C++">

```cpp
#include <filesystem>

#include <cuvs/neighbors/scann.hpp>

using namespace cuvs::neighbors::experimental;

raft::device_resources res;
auto dataset = load_device_dataset();

scann::index_params index_params;
auto index = scann::build(res, index_params, dataset);

std::filesystem::create_directories("/tmp/cuvs-scann-index");
scann::serialize(res, "/tmp/cuvs-scann-index", index);
```

</Tab>
</Tabs>

Serialization writes the files needed by OSS ScaNN, including partition centers, datapoint labels, PQ codebooks, quantized residuals, SOAR residuals, and optional bfloat16 reordering data.

### Searching an index

NVIDIA cuVS does not currently provide a SCaNN search API or SCaNN search parameters. To search a SCaNN index, serialize it with NVIDIA cuVS and load the generated files with OSS ScaNN.

### Loading a serialized index

NVIDIA cuVS does not currently expose a SCaNN deserialization API. The serialized directory is intended to be loaded by OSS ScaNN or another consumer that understands the OSS ScaNN file layout.

## How ScaNN works

SCaNN first partitions the dataset into leaves. A query only needs to consider promising leaves instead of the full dataset.

Next, SCaNN stores residual product quantization codes. A residual is the leftover difference between a vector and its assigned partition center. Product quantization compresses those residuals into compact codes.

SCaNN also computes SOAR labels. SOAR gives each vector another assignment that can help recover good candidates that would otherwise be missed near partition boundaries.

If `reordering_bf16` is enabled, NVIDIA cuVS also stores a bfloat16 copy of the dataset. OSS ScaNN can use that copy to rerank candidates with more accurate distances after the quantized first stage.

## When to use ScaNN

Use SCaNN when you want NVIDIA cuVS to build an OSS ScaNN-compatible index from C++ and you are comfortable with an experimental API.

Use SCaNN when partitioning plus quantization is a good fit for the dataset and you plan to search with OSS ScaNN.

Use IVF-Flat, IVF-PQ, CAGRA, brute-force, or Vamana instead when you need an NVIDIA cuVS search API, multi-language bindings, or a non-experimental API surface.

## Interoperability with OSS ScaNN

The SCaNN serializer writes a directory of files that OSS ScaNN can consume:

- `cuvs_metadata.bin`
- `centers.npy`
- `datapoint_to_token.npy`
- `pq_codebook.npy`
- `hashed_dataset.npy`
- `hashed_dataset_soar.npy`
- `bf16_dataset.npy`, when `reordering_bf16` is enabled

The implementation is experimental. Accuracy and performance are not currently guaranteed to match OSS ScaNN across releases.

## Using Filters

NVIDIA cuVS SCaNN does not expose a search API, so it does not expose NVIDIA cuVS filtering controls. Apply filtering in the OSS ScaNN search layer after loading the serialized index.

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` | Distance metric inherited from the common index parameters. |
| `metric_arg` | `2.0` | Extra argument for metrics that need one, such as Minkowski distance. |
| `n_leaves` | `1000` | Number of partition leaves. More leaves create smaller partitions, but increase partition metadata and tuning work. |
| `kmeans_n_rows_train` | `100000` | Number of rows sampled to train the partition centers. Reduce this if k-means training needs less memory. |
| `kmeans_n_iters` | `24` | Maximum number of k-means iterations used to train partition centers. |
| `partitioning_eta` | `1.0` | AVQ adjustment used while improving partition centers. Larger values change how strongly the adjustment favors dot-product preservation. |
| `soar_lambda` | `1.0` | SOAR spilling strength. Larger values can increase boundary coverage, but may change index balance. |
| `pq_dim` | `8` | Dimension of each product-quantization subspace. This must divide the dataset dimension. |
| `pq_bits` | `8` | Bits per PQ code. Supported values are `4` and `8`. |
| `pq_n_rows_train` | `100000` | Number of rows sampled to train PQ codebooks. The implementation caps this at `100000`. |
| `pq_train_iters` | `10` | Maximum number of iterations used to train PQ codebooks. |
| `reordering_bf16` | `false` | Stores a bfloat16 copy of the dataset for OSS ScaNN candidate reranking. |
| `reordering_noise_shaping_threshold` | `NaN` | Optional threshold for bfloat16 AVQ noise shaping. `NaN` disables the adjustment. |

## Tuning

Start with the defaults, then tune one part of the pipeline at a time.

Increase `n_leaves` when partitions are too large. Smaller partitions can reduce search work in OSS ScaNN, but too many leaves can make partition selection harder and increase metadata.

Tune `pq_dim` and `pq_bits` together. Smaller codes reduce memory, but can lower recall unless reranking has enough good candidates.

Use `soar_lambda` when recall suffers near partition boundaries. It controls the extra SOAR assignment that helps recover vectors that sit between leaves.

Enable `reordering_bf16` when final recall needs help and the extra host memory is acceptable.

## Memory footprint

SCaNN memory has three main parts: partition metadata, residual PQ codes, and optional reranking data. During build, NVIDIA cuVS also uses temporary training and batch workspaces. These estimates are derived from the current C++ storage layout and are intended for planning, not as exact allocator accounting.

To keep the formulas readable, this section uses short symbols. All estimates are in bytes. The examples convert bytes to MiB by dividing by `1024 * 1024`.

- `N`: Number of database vectors.
- `D`: Vector dimension.
- `B`: Bytes per input vector value. Use `4` for fp32.
- `L`: Number of partition leaves, or `n_leaves`.
- `P`: PQ subspace dimension, or `pq_dim`.
- `S`: Number of PQ subspaces, where `S = D / P`.
- `b`: Bits per PQ code, or `pq_bits`.
- `C`: Number of PQ clusters per subspace, where `C = 2^b`.
- `T_k`: K-means training rows, or `kmeans_n_rows_train`.
- `T_p`: PQ training rows, or `min(pq_n_rows_train, 100000)`.
- `Q_b`: Build batch size, currently `min(N, 65536)`.
- `R`: `1` when `reordering_bf16` is enabled, otherwise `0`.
- `S_idx`: Bytes per stored label, currently `sizeof(uint32_t)`.

The named terms in the formulas are also memory sizes:

- `centers_size`: Device memory for partition centers.
- `labels_size`: Device memory for normal and SOAR leaf labels.
- `pq_codebook_size`: Device memory for residual PQ codebooks.
- `residual_codes_size`: Host memory for normal and SOAR residual PQ codes.
- `bf16_dataset_size`: Optional host memory for bfloat16 reranking data.
- `*_peak`: Temporary peak memory for one build phase. Sequential phases are not added together.

### Scratch and maximum vectors

The formulas below include the largest visible build phases, but additional scratch can come from k-means, PQ training, SOAR workspace, allocator padding, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.25` for SCaNN build planning. If you can measure a representative smaller run, use:

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


For fixed `D`, `L`, `P`, `pq_bits`, and batch settings, most SCaNN storage terms are linear in `N`, while capped training terms can become fixed. Solve the full build formula or rewrite the dominant phase as:

$$
\text{peak\_without\_scratch}(N)
  = N \cdot B_{\text{per\_vector}} + B_{\text{fixed}}
$$

and estimate:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_vector}}}
  \right\rfloor
$$

Check device and host memory separately because SCaNN keeps residual codes and optional bfloat16 reranking data on host.

### Baseline memory after build

The baseline device memory kept by the SCaNN index is:

$$
\begin{aligned}
\text{centers\_size}
&= L \times D \times 4
\end{aligned}
$$

$$
\begin{aligned}
\text{labels\_size}
&= 2 \times N \times S_{\text{idx}}
\end{aligned}
$$

$$
\begin{aligned}
\text{pq\_codebook\_size}
&= C \times D \times 4
\end{aligned}
$$

$$
\begin{aligned}
\text{device\_index\_size}
&= \text{centers\_size} \\
&\quad + \text{labels\_size} \\
&\quad + \text{pq\_codebook\_size}
\end{aligned}
$$

The baseline host memory kept by the SCaNN index is:

$$
\begin{aligned}
\text{residual\_codes\_size}
&= 2 \times N \times S
\end{aligned}
$$

$$
\begin{aligned}
\text{bf16\_dataset\_size}
&= R \times N \times D \times 2
\end{aligned}
$$

$$
\begin{aligned}
\text{host\_index\_size}
&= \text{residual\_codes\_size} \\
&\quad + \text{bf16\_dataset\_size}
\end{aligned}
$$

The total index footprint is approximately:

$$
\begin{aligned}
\text{index\_size}
&\approx \text{device\_index\_size} \\
&\quad + \text{host\_index\_size}
\end{aligned}
$$

**Example** (`N = 1e6`, `D = 128`, `L = 1000`, `pq_dim = 8`, `pq_bits = 8`, `reordering_bf16 = false`):

- `centers_size = 512000 B = 0.49 MiB`
- `labels_size = 8000000 B = 7.63 MiB`
- `pq_codebook_size = 131072 B = 0.13 MiB`
- `residual_codes_size = 32000000 B = 30.52 MiB`
- `index_size = 40643072 B = 38.76 MiB`

### Build peak memory usage

SCaNN build runs in phases, so the temporary allocations below are not all held at once. The largest active phase usually dominates the extra build memory.

K-means training samples rows into device memory:

$$
\begin{aligned}
\text{kmeans\_training\_peak}
&= T_k \times D \times B
\end{aligned}
$$

PQ training samples residuals into device memory:

$$
\begin{aligned}
\text{pq\_training\_peak}
&= T_p \times D \times B
\end{aligned}
$$

Batch quantization stores normal residuals, SOAR residuals, and packed PQ codes for one build batch. Packed code width is `ceil(S * b / 8)` bytes per vector.

$$
\begin{aligned}
\text{packed\_code\_width}
&= \left\lceil \frac{S \times b}{8} \right\rceil
\end{aligned}
$$

$$
\begin{aligned}
\text{quantize\_batch\_peak}
&\approx 2 \times Q_b \times D \times B \\
&\quad + 2 \times Q_b
  \times \text{packed\_code\_width}
\end{aligned}
$$

When `pq_bits = 4`, NVIDIA cuVS also unpacks codes before copying them to the host index:

$$
\begin{aligned}
\text{unpack\_peak}
&= 2 \times Q_b \times S
\end{aligned}
$$

SOAR label computation uses a score matrix between one batch and all leaves. This is often the largest temporary in the quantization phase:

$$
\begin{aligned}
\text{soar\_workspace\_peak}
&\approx Q_b \times L \times 4 \\
&\quad + D \times L \times B
\end{aligned}
$$

If bfloat16 reordering is enabled, one device batch is quantized before being copied to host:

$$
\begin{aligned}
\text{bf16\_batch\_peak}
&= R \times Q_b \times D \times 2
\end{aligned}
$$

The overall build peak can be estimated as the dataset, the baseline device index, and the largest temporary phase:

$$
\begin{aligned}
\text{build\_peak}
&\approx N \times D \times B \\
&\quad + \text{device\_index\_size} \\
&\quad + \max\!\big(
  \text{kmeans\_training\_peak}, \\
&\qquad\qquad
  \text{pq\_training\_peak}, \\
&\qquad\qquad
  \text{quantize\_batch\_peak} \\
&\qquad\qquad
  + \text{unpack\_peak} \\
&\qquad\qquad
  + \text{soar\_workspace\_peak} \\
&\qquad\qquad
  + \text{bf16\_batch\_peak}
\big)
\end{aligned}
$$

### Serialization peak memory usage

Serialization writes host and device arrays to disk. It also creates a temporary device vector that combines normal labels and SOAR labels:

$$
\begin{aligned}
\text{combined\_labels\_size}
&= 2 \times N \times S_{\text{idx}}
\end{aligned}
$$

$$
\begin{aligned}
\text{serialize\_peak}
&\approx \text{index\_size} \\
&\quad + \text{combined\_labels\_size}
\end{aligned}
$$

### Search memory usage

NVIDIA cuVS does not currently search SCaNN indexes. Search memory depends on the OSS ScaNN configuration used after serialization.
