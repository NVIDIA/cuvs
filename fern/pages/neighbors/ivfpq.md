# IVF-PQ

IVF-PQ is a GPU-accelerated approximate nearest-neighbor index. It first partitions vectors into coarse clusters, also called inverted lists, then stores a compressed product-quantized code for each vector in the selected list.

Compared with IVF-Flat, IVF-PQ uses much less GPU memory because it stores compressed codes instead of full vectors. The trade-off is that search distances are approximate. For high recall, IVF-PQ is commonly paired with refinement, where the search returns more candidates than needed and the final top results are reranked with the original vectors.

IVF-PQ works well when the dataset is too large for full-precision storage on the GPU, some recall loss is acceptable, and the original vectors are available for optional reranking.

## Example API Usage

[C API](/api-reference/c-api-neighbors-ivf-pq) | [C++ API](/api-reference/cpp-api-neighbors-ivf-pq) | [Python API](/api-reference/python-api-neighbors-ivf-pq) | [Java IVF-PQ Params](/api-reference/java-api-com-nvidia-cuvs-cuvsivfpqparams) | [Rust API](/api-reference/rust-api-cuvs-neighbors-ivf-pq) | [Go API](/api-reference/go-api-ivf-pq)

Java currently exposes IVF-PQ parameter classes for CAGRA graph construction, not a standalone IVF-PQ index/search binding. The runnable standalone examples below cover C, C++, Python, Rust, and Go.

### Building an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_pq.h>

cuvsResources_t res;
cuvsIvfPqIndexParams_t index_params;
cuvsIvfPqIndex_t index;
DLManagedTensor *dataset;

// Populate tensor with data.
load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsIvfPqIndexParamsCreate(&index_params);
cuvsIvfPqIndexCreate(&index);

index_params->n_lists = 1024;
index_params->kmeans_trainset_fraction = 0.5;
index_params->pq_bits = 8;
index_params->pq_dim = 64;

cuvsIvfPqBuild(res, index_params, dataset, index);

// Keep index alive while you search, then destroy it when finished.
cuvsIvfPqIndexDestroy(index);
cuvsIvfPqIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_pq.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
ivf_pq::index_params index_params;

index_params.n_lists = 1024;
index_params.kmeans_trainset_fraction = 0.5;
index_params.pq_bits = 8;
index_params.pq_dim = 64;
index_params.metric = cuvs::distance::DistanceType::L2Expanded;

auto index = ivf_pq::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import ivf_pq

dataset = load_data()
index_params = ivf_pq.IndexParams(
    n_lists=1024,
    kmeans_trainset_fraction=0.5,
    pq_bits=8,
    pq_dim=64,
)

index = ivf_pq.build(index_params, dataset)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::ivf_pq::{Index, IndexParams};
use cuvs::{ManagedTensor, Resources, Result};

fn build_ivf_pq_index(dataset: &ndarray::Array2<f32>) -> Result<Index> {
    let res = Resources::new()?;
    let dataset = ManagedTensor::from(dataset).to_device(&res)?;
    let index_params = IndexParams::new()?
        .set_n_lists(1024)
        .set_pq_bits(8)
        .set_pq_dim(64);

    Index::build(&res, &index_params, dataset)
}
```

</Tab>
<Tab title="Go">

```go
package main

import (
	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/ivf_pq"
)

func buildIvfPqIndex(dataset cuvs.Tensor[float32]) (*ivf_pq.IvfPqIndex, error) {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return nil, err
	}

	indexParams, err := ivf_pq.CreateIndexParams()
	if err != nil {
		return nil, err
	}

	_, err = indexParams.SetNLists(1024)
	if err != nil {
		return nil, err
	}

	_, err = indexParams.SetPQBits(8)
	if err != nil {
		return nil, err
	}

	_, err = indexParams.SetPQDim(64)
	if err != nil {
		return nil, err
	}

	index, err := ivf_pq.CreateIndex(indexParams, &dataset)
	if err != nil {
		return nil, err
	}

	_, err = dataset.ToDevice(&resource)
	if err != nil {
		return nil, err
	}

	err = ivf_pq.BuildIndex(resource, indexParams, &dataset, index)
	return index, err
}
```

</Tab>
</Tabs>

### Extending an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_pq.h>

cuvsResources_t res;
cuvsIvfPqIndex_t index;
DLManagedTensor *new_vectors;
DLManagedTensor *new_indices;

load_additional_dataset(new_vectors);
load_additional_indices(new_indices);

cuvsResourcesCreate(&res);
cuvsIvfPqIndexCreate(&index);

// ... build or load index ...
cuvsIvfPqExtend(res, new_vectors, new_indices, index);

cuvsIvfPqIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_pq.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
ivf_pq::index_params index_params;
auto dataset = load_dataset();
auto new_vectors = load_additional_dataset();
auto new_indices = load_additional_indices<int64_t>();

auto index = ivf_pq::build(res, index_params, dataset);
index = ivf_pq::extend(res, new_vectors, new_indices.view(), index);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import ivf_pq

dataset = load_data()
new_vectors = load_additional_data()
new_indices = load_additional_indices()

index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
index = ivf_pq.extend(index, new_vectors, new_indices)
```

</Tab>
</Tabs>

See the [C](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqextend), [C++](/api-reference/cpp-api-neighbors-ivf-pq#neighbors-ivf-pq-extend), and [Python](/api-reference/python-api-neighbors-ivf-pq#extend) API references for the full signatures.

### Searching an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_pq.h>

cuvsResources_t res;
cuvsIvfPqSearchParams_t search_params;
cuvsIvfPqIndex_t index;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

// Populate tensors with query and output data.
load_queries(queries);
allocate_outputs(neighbors, distances);

cuvsResourcesCreate(&res);
cuvsIvfPqSearchParamsCreate(&search_params);
search_params->n_probes = 20;

// ... build or load index ...
cuvsIvfPqSearch(
    res,
    search_params,
    index,
    queries,
    neighbors,
    distances);

cuvsIvfPqSearchParamsDestroy(search_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_pq.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
ivf_pq::index<int64_t> index(res);
auto queries = load_queries();
auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
ivf_pq::search_params search_params;

search_params.n_probes = 20;

// ... build or load index ...
ivf_pq::search(
    res,
    search_params,
    index,
    queries,
    neighbors.view(),
    distances.view());
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import ivf_pq

queries = load_queries()
search_params = ivf_pq.SearchParams(n_probes=20)
k = 10

# ... build or load index ...
distances, neighbors = ivf_pq.search(search_params, index, queries, k)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::ivf_pq::SearchParams;
use cuvs::{ManagedTensor, Resources, Result};

fn search_ivf_pq_index(
    res: &Resources,
    index: &cuvs::ivf_pq::Index,
    queries: &ndarray::ArrayView2<f32>,
    k: usize,
) -> Result<()> {
    let n_queries = queries.shape()[0];
    let queries = ManagedTensor::from(queries).to_device(res)?;

    let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(res)?;

    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(res)?;

    let search_params = SearchParams::new()?.set_n_probes(20);
    index.search(res, &search_params, &queries, &neighbors, &distances)?;

    distances.to_host(res, &mut distances_host)?;
    neighbors.to_host(res, &mut neighbors_host)?;

    Ok(())
}
```

</Tab>
<Tab title="Go">

```go
searchParams, err := ivf_pq.CreateSearchParams()
if err != nil {
	return err
}
defer searchParams.Close()

_, err = searchParams.SetNProbes(20)
if err != nil {
	return err
}

_, err = queries.ToDevice(&resource)
if err != nil {
	return err
}

err = ivf_pq.SearchIndex(
	resource,
	searchParams,
	index,
	&queries,
	&neighbors,
	&distances,
)
if err != nil {
	return err
}

_, err = neighbors.ToHost(&resource)
if err != nil {
	return err
}

_, err = distances.ToHost(&resource)
if err != nil {
	return err
}
```

</Tab>
</Tabs>

### Saving and loading an index

Serialize an IVF-PQ index when you want to reuse trained coarse clusters, PQ codebooks, and compressed lists without rebuilding the index.

Java currently exposes IVF-PQ parameter classes for CAGRA graph construction, not a standalone IVF-PQ index/search binding. Rust and Go currently expose IVF-PQ build/search wrappers, but not IVF-PQ save/load wrappers.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/ivf_pq.h>

cuvsResources_t res;
cuvsIvfPqIndex_t index;
cuvsIvfPqIndex_t loaded_index;

cuvsResourcesCreate(&res);
cuvsIvfPqIndexCreate(&index);
cuvsIvfPqIndexCreate(&loaded_index);

// ... build index ...
cuvsIvfPqSerialize(res, "/tmp/cuvs-ivf-pq.bin", index);
cuvsIvfPqDeserialize(res, "/tmp/cuvs-ivf-pq.bin", loaded_index);

cuvsIvfPqIndexDestroy(loaded_index);
cuvsIvfPqIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_pq.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
ivf_pq::index_params index_params;
auto index = ivf_pq::build(res, index_params, dataset);

ivf_pq::serialize(res, "/tmp/cuvs-ivf-pq.bin", index);

ivf_pq::index<int64_t> loaded_index(res);
ivf_pq::deserialize(res, "/tmp/cuvs-ivf-pq.bin", &loaded_index);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import ivf_pq

dataset = load_data()
index = ivf_pq.build(ivf_pq.IndexParams(), dataset)

ivf_pq.save("/tmp/cuvs-ivf-pq.bin", index, include_dataset=True)
loaded_index = ivf_pq.load("/tmp/cuvs-ivf-pq.bin")
```

</Tab>
</Tabs>

## How IVF-PQ works

IVF-PQ combines two ideas:

1. IVF partitions the dataset into `n_lists` coarse clusters. During search, each query visits only the closest `n_probes` lists.
2. PQ compresses each vector into a short code. Instead of comparing the query to every original vector value, search compares the query to precomputed codebook entries.

Think of each vector as a long row of numbers. PQ splits that row into smaller chunks, gives each chunk a short code, and stores the codes. At search time, NVIDIA cuVS uses lookup tables to score those codes quickly.

The compression is lossy, so IVF-PQ trades some recall for a smaller and faster index. Refinement can recover much of that recall when the original vectors are still available.

## When to use IVF-PQ

Use IVF-PQ when GPU memory is the main limit and IVF-Flat would store too much full-precision data.

IVF-PQ is often a good fit for large datasets, high-throughput candidate generation, and workflows that can rerank candidates with exact distances afterward.

Avoid IVF-PQ when exact distances are required for every result, when recall cannot tolerate compression error, or when the original vectors are unavailable but high recall depends on refinement.

## Refinement and reranking

IVF-PQ search returns approximate distances from compressed vectors. A common pattern is to ask IVF-PQ for `K0` candidates, where `K0` is larger than the final `K`, then rerank those candidates with exact distances against the original dataset.

For example, search for `K0 = 4 * K`, refine the candidates, and keep the best `K`. This needs the original vectors, but it can improve recall substantially without storing full vectors inside the IVF-PQ index.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/refine.h>

int64_t k = 10;
int64_t k0 = 4 * k;

DLManagedTensor *dataset;
DLManagedTensor *queries;
DLManagedTensor *candidate_neighbors;
DLManagedTensor *candidate_distances;
DLManagedTensor *refined_neighbors;
DLManagedTensor *refined_distances;

// Allocate candidate output as [n_queries, k0].
allocate_outputs(candidate_neighbors, candidate_distances);

cuvsIvfPqSearch(
    res,
    search_params,
    index,
    queries,
    candidate_neighbors,
    candidate_distances);

// Allocate refined output as [n_queries, k].
allocate_outputs(refined_neighbors, refined_distances);

cuvsRefine(
    res,
    dataset,
    queries,
    candidate_neighbors,
    index_params->metric,
    refined_neighbors,
    refined_distances);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

using namespace cuvs::neighbors;

int64_t k = 10;
int64_t k0 = 4 * k;

auto candidate_neighbors =
    raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k0);
auto candidate_distances =
    raft::make_device_matrix<float, int64_t>(res, n_queries, k0);

ivf_pq::search(
    res,
    search_params,
    index,
    queries,
    candidate_neighbors.view(),
    candidate_distances.view());

auto refined_neighbors =
    raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
auto refined_distances =
    raft::make_device_matrix<float, int64_t>(res, n_queries, k);

refine(
    res,
    dataset,
    queries,
    raft::make_const_mdspan(candidate_neighbors.view()),
    refined_neighbors.view(),
    refined_distances.view(),
    index.metric());
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import ivf_pq, refine

k = 10
k0 = 4 * k

candidate_distances, candidates = ivf_pq.search(
    search_params,
    index,
    queries,
    k0,
)

distances, neighbors = refine(
    dataset,
    queries,
    candidates,
    k,
    metric="sqeuclidean",
)
```

</Tab>
</Tabs>

Rust and Go currently expose standalone IVF-PQ search without a separate refinement wrapper.

## Using Filters

IVF-PQ C++ search supports sample filters. Filtering happens only inside the lists selected by `n_probes`, so a filtered search can miss an allowed vector if that vector lives in a list that was not probed.

If filtered recall matters, increase `n_probes`, evaluate with representative filters, or use brute-force when the filter is expected to remove most vectors. C, Python, Rust, and Go currently expose standalone IVF-PQ search without a filter argument, and Java does not currently expose a standalone IVF-PQ binding.

```cpp
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

using namespace cuvs::neighbors;
using indexing_dtype = int64_t;

raft::device_resources res;
ivf_pq::index<indexing_dtype> index(res);
ivf_pq::search_params search_params;
auto queries = load_queries();

auto neighbors =
    raft::make_device_matrix<indexing_dtype, indexing_dtype>(res, n_queries, k);
auto distances =
    raft::make_device_matrix<float, indexing_dtype>(res, n_queries, k);

// Load a list of all samples that should be allowed.
std::vector<indexing_dtype> allowed_indices_host = get_allowed_indices();
auto allowed_indices_device =
    raft::make_device_vector<indexing_dtype, indexing_dtype>(
        res,
        allowed_indices_host.size());

raft::copy(
    allowed_indices_device.data_handle(),
    allowed_indices_host.data(),
    allowed_indices_host.size(),
    raft::resource::get_cuda_stream(res));

cuvs::core::bitset<uint32_t, indexing_dtype> allowed_indices_bitset(
    res,
    allowed_indices_device.view(),
    index.size());
auto bitset_filter =
    filtering::bitset_filter(allowed_indices_bitset.view());

ivf_pq::search(
    res,
    search_params,
    index,
    queries,
    neighbors.view(),
    distances.view(),
    bitset_filter);
```

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / `sqeuclidean` | Distance metric used for cluster training, PQ scoring, and search. |
| `metric_arg` | `2.0` | Extra argument for metrics that need one, such as Minkowski distance. |
| `n_lists` | `1024` | Number of coarse clusters, or inverted lists. More lists make each list smaller, but may require more probes to maintain recall. |
| `kmeans_n_iters` | `20` | Maximum number of k-means iterations used to train the coarse list centers. |
| `kmeans_trainset_fraction` | `0.5` | Fraction of the dataset used for coarse k-means training. Reducing this can speed up build on very large datasets. |
| `pq_bits` | `8` | Number of bits used for each PQ code. Smaller values reduce memory and lookup-table size, but can lower recall. |
| `pq_dim` | `0` | Number of PQ code dimensions. `0` lets NVIDIA cuVS choose a heuristic value. `pq_dim * pq_bits` must be a multiple of 8. |
| `codebook_kind` | `per_subspace` | Trains one codebook per PQ subspace, or separate codebooks per cluster with `per_cluster`. Per-cluster codebooks can improve recall but use more memory. |
| `codes_layout` | `interleaved` | Memory layout for compressed list data. `interleaved` is optimized for GPU search. `flat` stores each vector code contiguously. |
| `force_random_rotation` | `False` | Applies a random rotation even when the input dimension is already compatible with `pq_dim`. Rotation is always needed when the dimension must be padded to a multiple of `pq_dim`. |
| `conservative_memory_allocation` | `False` | Uses tighter per-list allocations. The default overallocates lists to reduce reallocations during repeated `extend` calls. |
| `add_data_on_build` | `True` | Adds vectors to the index during build. Set to `False` to train the quantizers first and add vectors later with `extend`. |
| `max_train_points_per_pq_code` | `256` | Maximum training points used per PQ code during codebook training. Higher values can improve codebooks but increase build time. |

### Search parameters

| Name | Default | Description |
| --- | --- | --- |
| `n_probes` | `20` | Number of closest lists to scan for each query. This is the main recall and latency knob. |
| `lut_dtype` | `CUDA_R_32F` / `float32` | Data type used for PQ lookup tables. Lower precision can reduce shared memory use and improve speed, with possible recall loss. |
| `internal_distance_dtype` | `CUDA_R_32F` / `float32` | Data type used for search-time distance accumulation. `CUDA_R_16F` can improve performance when memory access dominates. |
| `preferred_shmem_carveout` | `1.0` | Hint for the fraction of unified L1/shared memory used as shared memory. This is a low-level performance knob. |
| `coarse_search_dtype` | `CUDA_R_32F` / `float32` | Data type used when finding the closest coarse clusters. Lower precision is faster but can reduce recall. |
| `max_internal_batch_size` | `4096` | Internal search batch size. Larger batches can improve GPU utilization but require more temporary memory. |

Filters are passed as search function arguments in bindings that expose filtered search, not as fields in `ivf_pq::search_params`.

## Tuning

Start with `n_lists`. A common target is roughly 1,000 to 10,000 vectors per list. More lists make each list smaller, but search usually needs a higher `n_probes` value to maintain recall.

Tune `n_probes` next. Increasing `n_probes` usually improves recall because each query scans more lists, but it also increases latency.

Then tune compression. Lower `pq_bits` and lower `pq_dim` reduce memory and can speed up search, but they also make each stored vector less precise. If recall drops too much, increase `pq_bits`, increase `pq_dim`, use `per_cluster` codebooks, or add refinement.

Use lower-precision `lut_dtype`, `internal_distance_dtype`, or `coarse_search_dtype` only after the basic recall and latency balance is acceptable. These options can improve throughput, but they add another source of approximation.

## Memory footprint

IVF-PQ memory has four main parts: compressed vector codes, source row IDs, coarse cluster centers, and PQ codebooks. The index is much smaller than IVF-Flat because it stores `pq_bits`-wide codes rather than full vectors.

To keep the formulas readable, this section uses short symbols. All estimates are in bytes. The examples convert bytes to MiB by dividing by `1024 * 1024`.

- `N`: Number of database vectors, or rows in the dataset being indexed.
- `D`: Original vector dimension, or number of values in each vector.
- `D_rot`: Rotated and padded dimension used internally by PQ. This is a multiple of `M`.
- `B`: Bytes stored for each original vector value. Use `4` for fp32, `2` for fp16, or `1` for int8 and uint8.
- `L`: Number of inverted lists. This is the `n_lists` build parameter.
- `M`: Number of PQ code dimensions. This is the `pq_dim` build parameter.
- `b`: Bits per PQ code. This is the `pq_bits` build parameter.
- `C`: Number of entries in each PQ codebook, equal to `2^b`.
- `S_idx`: Bytes per source row ID. This is `sizeof(IdxT)`, usually `8` for `int64_t`.
- `N_cap`: Total allocated list capacity across all lists after padding and over-allocation.
- `Q`: Query batch size, or number of query vectors processed together.
- `K`: Search result count, or the requested `k` nearest neighbors per query.
- `K0`: Candidate count used before refinement, where `K0 >= K`.
- `P`: Number of lists probed during search. This is the `n_probes` search parameter.
- `F`: Training-set fraction. This is the `kmeans_trainset_fraction` build parameter.
- `B_lut`: Bytes per lookup-table value. This depends on `lut_dtype`.

The named terms in the formulas are also memory sizes:

- `encoded_data_size`: Device memory used by compressed PQ codes stored inside IVF lists.
- `list_index_size`: Device memory used by source row IDs stored beside list codes.
- `centers_size`: Device memory used by the `L` coarse cluster centers.
- `rotation_size`: Device memory used by the optional random rotation matrix and rotated centers.
- `codebook_size`: Device memory used by PQ codebooks.
- `index_size`: Approximate device memory used by the trained IVF-PQ index.
- `query_size`: Device memory for the current query batch.
- `result_size`: Device memory for neighbor IDs and distances returned for the current query batch.
- `workspace_size`: Temporary search workspace.
- `refinement_dataset_size`: Memory needed for original vectors when refinement reranks candidates.

### Scratch and maximum vectors

The formulas below show the major persistent buffers and search workspaces. Additional scratch comes from k-means training, PQ training, list allocation padding, allocator fragmentation, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.20` for build estimates and `H = 0.10` for search estimates. If you can measure a representative smaller run, use:

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


IVF-PQ capacity depends on list padding and the selected code layout. Estimate `N_cap` first for the default interleaved layout. Then solve the selected peak formula:

$$
\begin{aligned}
\text{peak\_without\_scratch}(N)
  =&\ \text{encoded\_data\_size}(N_{\text{cap}}) \\
  &+ N \cdot B_{\text{per\_input\_vector}}
   + B_{\text{fixed}}
\end{aligned}
$$

For a rough maximum-vector estimate, try increasing `N` values until `peak_without_scratch(N) <= M_usable` no longer holds. If using the `flat` layout and no refinement dataset, most terms are linear in `N`, so the shortcut is:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_vector}}}
  \right\rfloor
$$

### List capacity

The default IVF-PQ list layout is `interleaved`. It pads each list for efficient GPU memory access. The minimum alignment is 32 rows. With the default allocation strategy, lists may be overallocated up to 1024-row chunks to make future `extend` calls cheaper. With `conservative_memory_allocation = true`, the maximum alignment is 32 rows.

For each list `i`, let `n_i` be the number of vectors assigned to that list.

$$
\begin{aligned}
A_{\text{min}} &= 32
\end{aligned}
$$

$$
\begin{aligned}
A_{\text{max}}
&=
\begin{cases}
32, & \text{conservative allocation} \\
1024, & \text{default allocation}
\end{cases}
\end{aligned}
$$

For the default `interleaved` layout, the implementation chooses a capacity for each list based on `A_min` and `A_max`:

$$
\begin{aligned}
\operatorname{capacity}(n_i)
&=
\begin{cases}
\min\!\big(
  \operatorname{nextpow2}
  (\max(n_i, A_{\text{min}})),
  A_{\text{max}}
\big),
& n_i < A_{\text{max}} \\
\operatorname{round\_up}(n_i, A_{\text{max}}),
& n_i \ge A_{\text{max}}
\end{cases}
\end{aligned}
$$

The total allocated list capacity is:

$$
\begin{aligned}
N_{\text{cap}}
&= \sum_{i=1}^{L}
  \operatorname{capacity}(n_i)
\end{aligned}
$$

For balanced lists, a simple approximation is:

$$
\begin{aligned}
N_{\text{cap}}
&\approx L \times
  \operatorname{capacity}
  \left(\frac{N}{L}\right)
\end{aligned}
$$

The `flat` code layout stores exactly one contiguous code per vector, so its list capacity is approximately `N` instead of `N_cap`. Use the exact list sizes when estimating tightly.

### Baseline memory after build

For the `flat` code layout, each compressed vector uses:

$$
\begin{aligned}
\text{bytes\_per\_code}
&=
\left\lceil
  \frac{M \times b}{8}
\right\rceil
\end{aligned}
$$

The flat encoded data size is:

$$
\begin{aligned}
\text{encoded\_data\_size}_{\text{flat}}
&=
N \times \text{bytes\_per\_code}
\end{aligned}
$$

For the default `interleaved` code layout, codes are stored in 32-row groups and 16-byte vectorized chunks. Let:

$$
\begin{aligned}
C_{\text{chunk}}
&=
\frac{16 \times 8}{b}
\end{aligned}
$$

Then the encoded data size is approximately:

$$
\begin{aligned}
\text{encoded\_data\_size}_{\text{interleaved}}
&=
\sum_{i=1}^{L}
\left\lceil
  \frac{\operatorname{capacity}(n_i)}{32}
\right\rceil \\
&\quad \times
\left\lceil
  \frac{M}{C_{\text{chunk}}}
\right\rceil
\times 32 \times 16
\end{aligned}
$$

Source row IDs are stored beside the compressed codes:

$$
\begin{aligned}
\text{list\_index\_size}
&= N_{\text{cap}} \times S_{\text{idx}}
\end{aligned}
$$

Coarse centers are stored in full precision:

$$
\begin{aligned}
\text{centers\_size}
&= L \times D
  \times \operatorname{sizeof}(\mathrm{float})
\end{aligned}
$$

When a rotation is needed, the rotated centers and rotation matrix add:

$$
\begin{aligned}
\text{rotation\_size}
&\approx
  L \times D_{\text{rot}}
  \times \operatorname{sizeof}(\mathrm{float}) \\
&\quad +
  D_{\text{rot}} \times D
  \times \operatorname{sizeof}(\mathrm{float})
\end{aligned}
$$

For `per_subspace` codebooks:

$$
\begin{aligned}
\text{codebook\_size}_{\text{per\_subspace}}
&=
M \times
\frac{D_{\text{rot}}}{M}
\times C
\times \operatorname{sizeof}(\mathrm{float})
\end{aligned}
$$

For `per_cluster` codebooks:

$$
\begin{aligned}
\text{codebook\_size}_{\text{per\_cluster}}
&=
L \times
\frac{D_{\text{rot}}}{M}
\times C
\times \operatorname{sizeof}(\mathrm{float})
\end{aligned}
$$

The baseline index footprint is:

$$
\begin{aligned}
\text{index\_size}
&\approx
  \text{encoded\_data\_size} \\
&\quad + \text{list\_index\_size} \\
&\quad + \text{centers\_size} \\
&\quad + \text{rotation\_size} \\
&\quad + \text{codebook\_size} \\
&\quad + \text{small list metadata}
\end{aligned}
$$

**Example** (`N = 1e6`, `D = 128`, `D_rot = 128`, `L = 1024`, `M = 64`, `b = 8`, balanced lists, default interleaved layout, `IdxT = int64`, `per_subspace` codebooks):

- `N / L = 976.56`, so `N_cap ~= 1024 * 1024 = 1,048,576`
- `encoded_data_size ~= 67,108,864 B = 64.00 MiB`
- `list_index_size = 8,388,608 B = 8.00 MiB`
- `centers_size = 524,288 B = 0.50 MiB`
- `rotation_size ~= 589,824 B = 0.56 MiB`
- `codebook_size = 131,072 B = 0.13 MiB`
- `index_size ~= 73.19 MiB + small metadata`

The original dataset would be `1e6 * 128 * 4 = 488.28 MiB` in fp32, so IVF-PQ stores the searchable index much more compactly.

### Refinement memory

Refinement needs the original vectors in addition to the IVF-PQ index:

$$
\begin{aligned}
\text{refinement\_dataset\_size}
&= N \times D \times B
\end{aligned}
$$

It also needs candidate IDs from the approximate IVF-PQ search:

$$
\begin{aligned}
\text{candidate\_size}
&= Q \times K0 \times S_{\text{idx}}
\end{aligned}
$$

When the original dataset is too large to keep on the GPU, refinement can be staged or performed where the original vectors are available.

### Build peak memory usage

Build trains coarse centers, trains PQ codebooks, and fills the lists when `add_data_on_build = true`. Peak memory depends on whether the input dataset is already on device and how the workspace memory resource is configured.

The k-means training sample size is:

$$
\begin{aligned}
N_{\text{train}}
&= F \times N
\end{aligned}
$$

A practical build estimate is:

$$
\begin{aligned}
\text{training\_sample\_size}
&= N_{\text{train}} \times D \times B
\end{aligned}
$$

$$
\begin{aligned}
\text{training\_labels\_size}
&= N_{\text{train}}
  \times \operatorname{sizeof}(\mathrm{uint32\_t})
\end{aligned}
$$

The PQ training workspace depends on `pq_bits`, `pq_dim`, `codebook_kind`, and `max_train_points_per_pq_code`. A useful estimate for the number of training vectors per codebook is:

$$
\begin{aligned}
N_{\text{pq\_train}}
&\lesssim
C \times \text{max\_train\_points\_per\_pq\_code}
\end{aligned}
$$

The total build peak is approximately:

$$
\begin{aligned}
\text{build\_peak}
&\approx
  \text{input\_dataset\_size} \\
&\quad + \text{training\_sample\_size} \\
&\quad + \text{training\_labels\_size} \\
&\quad + \text{centers\_size} \\
&\quad + \text{rotation\_size} \\
&\quad + \text{codebook\_training\_workspace} \\
&\quad + \text{index\_size}
\end{aligned}
$$

When the workspace memory resource does not have enough room for the training set and labels, IVF-PQ build can fall back to managed memory for those temporary allocations.

### Search peak memory usage

IVF-PQ search requires the compressed index to be resident on the GPU. Search also needs query vectors, output buffers, PQ lookup tables, and temporary workspace.

$$
\begin{aligned}
\text{query\_size}
&= Q \times D \times B
\end{aligned}
$$

$$
\begin{aligned}
\text{result\_size}
&= Q \times K \\
&\quad \times
  \big(S_{\text{idx}}
  + \operatorname{sizeof}(\mathrm{float})\big)
\end{aligned}
$$

A useful lookup-table estimate is:

$$
\begin{aligned}
P_{\text{codebook}}
&=
\begin{cases}
1, & \text{per-subspace codebooks} \\
P, & \text{per-cluster codebooks}
\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
\text{lut\_size}
&\approx
Q \times P_{\text{codebook}}
\times M \times C \times B_{\text{lut}}
\end{aligned}
$$

A practical workspace estimate is:

$$
\begin{aligned}
\text{workspace\_size}
&\approx
  \text{lut\_size} \\
&\quad +
Q \times
\Big[
  \big(L + 1 + P \times (K + 1)\big)
  \times \operatorname{sizeof}(\mathrm{float}) \\
&\qquad +
  P \times K \times S_{\text{idx}}
\Big]
\end{aligned}
$$

The total search memory is:

$$
\begin{aligned}
\text{search\_memory}
&\approx
  \text{index\_size} \\
&\quad + \text{query\_size} \\
&\quad + \text{result\_size} \\
&\quad + \text{workspace\_size}
\end{aligned}
$$

For many lists, use a pool memory resource when possible. Each IVF list is allocated separately, and a pool allocator helps avoid large per-allocation overhead from the underlying device memory resource.
