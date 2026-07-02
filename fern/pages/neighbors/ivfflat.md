# IVF-Flat

IVF-Flat is a GPU-accelerated approximate nearest-neighbor index. It partitions vectors into coarse clusters, also called inverted lists, and stores the original vectors inside those lists.

During search, IVF-Flat first finds the closest lists, then scans only those lists. This can be much faster than brute-force because search does not need to compare each query with every vector in the dataset.

IVF-Flat works well when the index fits in GPU memory, exact recall is not required, and partitioning the dataset gives a useful speedup.

## Example API Usage

[C API](/api-reference/c-api-neighbors-ivf-flat) | [C++ API](/api-reference/cpp-api-neighbors-ivf-flat) | [Python API](/api-reference/python-api-neighbors-ivf-flat) | [Rust API](/api-reference/rust-api-cuvs-ivf-flat) | [Go API](/api-reference/go-api-ivf-flat)

### Building an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/ivf_flat.h>

cuvsResources_t res;
cuvsIvfFlatIndexParams_t index_params;
cuvsIvfFlatIndex_t index;
DLManagedTensor *dataset;

// Populate tensor with data.
load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsIvfFlatIndexParamsCreate(&index_params);
cuvsIvfFlatIndexCreate(&index);

index_params->n_lists = 1024;
index_params->kmeans_trainset_fraction = 0.5;

cuvsIvfFlatBuild(res, index_params, dataset, index);

cuvsIvfFlatIndexDestroy(index);
cuvsIvfFlatIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_flat.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
ivf_flat::index_params index_params;

index_params.n_lists = 1024;
index_params.kmeans_trainset_fraction = 0.5;
index_params.metric = cuvs::distance::DistanceType::L2Expanded;

auto index = ivf_flat::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import ivf_flat

dataset = load_data()
index_params = ivf_flat.IndexParams(
    n_lists=1024,
    kmeans_trainset_fraction=0.5,
)

index = ivf_flat.build(index_params, dataset)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::ivf_flat::{Index, IndexParams};
use cuvs::{ManagedTensor, Resources, Result};

fn build_ivf_flat_index(dataset: &ndarray::Array2<f32>) -> Result<Index> {
    let res = Resources::new()?;
    let dataset = ManagedTensor::from(dataset).to_device(&res)?;
    let index_params = IndexParams::new()?.set_n_lists(1024);

    Index::build(&res, &index_params, dataset)
}
```

</Tab>
<Tab title="Go">

```go
package main

import (
	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/ivf_flat"
)

func buildIvfFlatIndex(dataset cuvs.Tensor[float32]) (*ivf_flat.IvfFlatIndex, error) {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return nil, err
	}

	indexParams, err := ivf_flat.CreateIndexParams()
	if err != nil {
		return nil, err
	}

	_, err = indexParams.SetNLists(1024)
	if err != nil {
		return nil, err
	}

	index, err := ivf_flat.CreateIndex(indexParams, &dataset)
	if err != nil {
		return nil, err
	}

	_, err = dataset.ToDevice(&resource)
	if err != nil {
		return nil, err
	}

	err = ivf_flat.BuildIndex(resource, indexParams, &dataset, index)
	return index, err
}
```

</Tab>
</Tabs>

### Extending an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/ivf_flat.h>

cuvsResources_t res;
cuvsIvfFlatIndex_t index;
DLManagedTensor *new_vectors;
DLManagedTensor *new_indices;

load_additional_dataset(new_vectors);
load_additional_indices(new_indices);

cuvsResourcesCreate(&res);
cuvsIvfFlatIndexCreate(&index);

// ... build or load index ...
cuvsIvfFlatExtend(res, new_vectors, new_indices, index);

cuvsIvfFlatIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_flat.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
ivf_flat::index_params index_params;
auto dataset = load_dataset();
auto new_vectors = load_additional_dataset();
auto new_indices = load_additional_indices<int64_t>();

auto index = ivf_flat::build(res, index_params, dataset);
index = ivf_flat::extend(res, new_vectors, new_indices.view(), index);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import ivf_flat

dataset = load_data()
new_vectors = load_additional_data()
new_indices = load_additional_indices()

index = ivf_flat.build(ivf_flat.IndexParams(), dataset)
index = ivf_flat.extend(index, new_vectors, new_indices)
```

</Tab>
</Tabs>

See the [C](/api-reference/c-api-neighbors-ivf-flat#cuvsivfflatextend), [C++](/api-reference/cpp-api-neighbors-ivf-flat#neighbors-ivf-flat-extend), and [Python](/api-reference/python-api-neighbors-ivf-flat#extend) API references for the full signatures.

### Searching an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/ivf_flat.h>

cuvsResources_t res;
cuvsIvfFlatSearchParams_t search_params;
cuvsIvfFlatIndex_t index;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

// Populate tensor with data.
load_queries(queries);
allocate_outputs(neighbors, distances);

cuvsResourcesCreate(&res);
cuvsIvfFlatSearchParamsCreate(&search_params);
search_params->n_probes = 20;

// ... build or load index ...
cuvsFilter filter = {0, NO_FILTER};
cuvsIvfFlatSearch(
    res,
    search_params,
    index,
    queries,
    neighbors,
    distances,
    filter);

cuvsIvfFlatSearchParamsDestroy(search_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_flat.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
ivf_flat::index<float, int64_t> index(res);
auto queries = load_queries();
auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
ivf_flat::search_params search_params;

search_params.n_probes = 20;

// ... build or load index ...
ivf_flat::search(
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
from cuvs.neighbors import ivf_flat

queries = load_queries()
search_params = ivf_flat.SearchParams(n_probes=20)
k = 10

# ... build or load index ...
distances, neighbors = ivf_flat.search(search_params, index, queries, k)
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::ivf_flat::SearchParams;
use cuvs::{ManagedTensor, Resources, Result};

fn search_ivf_flat_index(
    res: &Resources,
    index: &cuvs::ivf_flat::Index,
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
searchParams, err := ivf_flat.CreateSearchParams()
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

err = ivf_flat.SearchIndex(resource, searchParams, index, &queries, &neighbors, &distances)
if err != nil {
	return err
}

_, err = neighbors.ToHost(&resource)
if err != nil {
	return err
}

_, err = distances.ToHost(&resource)
return err
```

</Tab>
</Tabs>

### Saving and loading an index

Serialize an IVF-Flat index when you want to reuse trained lists and stored vectors without rebuilding the index.

Java does not currently expose a standalone IVF-Flat binding. Rust and Go currently expose build/search wrappers, but not IVF-Flat save/load wrappers.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/ivf_flat.h>

cuvsResources_t res;
cuvsIvfFlatIndex_t index;
cuvsIvfFlatIndex_t loaded_index;

cuvsResourcesCreate(&res);
cuvsIvfFlatIndexCreate(&index);
cuvsIvfFlatIndexCreate(&loaded_index);

// ... build index ...
cuvsIvfFlatSerialize(res, "/tmp/cuvs-ivf-flat.bin", index);
cuvsIvfFlatDeserialize(res, "/tmp/cuvs-ivf-flat.bin", loaded_index);

cuvsIvfFlatIndexDestroy(loaded_index);
cuvsIvfFlatIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/ivf_flat.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
ivf_flat::index_params index_params;
auto index = ivf_flat::build(res, index_params, dataset);

ivf_flat::serialize(res, "/tmp/cuvs-ivf-flat.bin", index);

ivf_flat::index<float, int64_t> loaded_index(res);
ivf_flat::deserialize(res, "/tmp/cuvs-ivf-flat.bin", &loaded_index);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import ivf_flat

dataset = load_data()
index = ivf_flat.build(ivf_flat.IndexParams(), dataset)

ivf_flat.save("/tmp/cuvs-ivf-flat.bin", index, include_dataset=True)
loaded_index = ivf_flat.load("/tmp/cuvs-ivf-flat.bin")
```

</Tab>
</Tabs>

## How IVF-Flat works

IVF-Flat has two main steps.

First, build trains coarse cluster centers with k-means. Each vector is assigned to the closest center, and vectors assigned to the same center are stored together in one inverted list.

Second, search compares each query with the cluster centers, chooses the closest `n_probes` lists, and scans the full-precision vectors inside those lists.

The search is approximate because vectors in lists that are not probed are skipped. Increasing `n_probes` searches more lists and usually improves recall, but it also increases latency.

## When to use IVF-Flat

Use IVF-Flat when brute-force is too slow, but you still want to store full-precision vectors.

Use IVF-Flat when the dataset is large enough that partitioning helps search avoid most vectors.

Use IVF-Flat when you can tune recall and throughput by changing how many lists are searched.

Use brute-force instead when exact results are required or the dataset is small enough that scanning every vector is already fast enough.

Use IVF-PQ instead when the full-precision IVF-Flat index is too large for device memory and some compression loss is acceptable.

## IVF-SQ and scalar quantization

IVF-SQ follows the same partition-first search pattern as IVF-Flat, but stores scalar-quantized vector values inside each partition. NVIDIA cuVS exposes scalar quantization as a preprocessing API rather than a separate standalone IVF-SQ index API.

Use this pattern when full-precision IVF-Flat storage is too large, but you want simpler compression than IVF-PQ. Scalar quantization reduces memory bandwidth and index size while usually requiring less tuning than product quantization.

## Using Filters

IVF-Flat supports filtered search, but filtering happens only inside the lists selected by `n_probes`.

This means a filtered IVF-Flat search can miss an allowed vector if that vector lives in a list that was not probed. If filtered recall matters, increase `n_probes`, use a filter-aware evaluation set, or use brute-force when the filter is expected to remove most vectors.

The examples below use a bitset filter. A bit value of `1` means a vector is allowed; a bit value of `0` means it is filtered out.

These examples cover the IVF-Flat bindings that currently expose filters. Rust and Go currently expose IVF-Flat search without a filter argument, and Java does not currently expose a standalone IVF-Flat binding.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/ivf_flat.h>

cuvsResources_t res;
cuvsIvfFlatIndexParams_t index_params;
cuvsIvfFlatSearchParams_t search_params;
cuvsIvfFlatIndex_t index;
DLManagedTensor *dataset;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

cuvsResourcesCreate(&res);
cuvsIvfFlatIndexParamsCreate(&index_params);
cuvsIvfFlatSearchParamsCreate(&search_params);
cuvsIvfFlatIndexCreate(&index);

// Populate DLPack tensors with dataset, query, and output data.
load_dataset(dataset);
load_queries(queries);
allocate_outputs(neighbors, distances);

cuvsIvfFlatBuild(res, index_params, dataset, index);

// Create a device uint32 bitset with one bit per indexed vector. Bit 1 means
// allowed; bit 0 means filtered out.
DLManagedTensor *bitset = make_device_bitset(allowed_indices, n_vectors);

cuvsFilter filter;
filter.type = BITSET;
filter.addr = (uintptr_t)bitset;

cuvsIvfFlatSearch(
    res,
    search_params,
    index,
    queries,
    neighbors,
    distances,
    filter);

cuvsIvfFlatIndexDestroy(index);
cuvsIvfFlatSearchParamsDestroy(search_params);
cuvsIvfFlatIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>

using namespace cuvs::neighbors;
using indexing_dtype = int64_t;

raft::device_resources res;
ivf_flat::index<float, indexing_dtype> index(res);
ivf_flat::search_params search_params;
auto queries = load_queries();

auto neighbors = raft::make_device_matrix<indexing_dtype, indexing_dtype>(res, n_queries, k);
auto distances = raft::make_device_matrix<float, indexing_dtype>(res, n_queries, k);

// Load a list of all samples that should be allowed.
std::vector<indexing_dtype> allowed_indices_host = get_allowed_indices();
auto allowed_indices_device =
    raft::make_device_vector<indexing_dtype, indexing_dtype>(
        res,
        allowed_indices_host.size());

raft::copy(allowed_indices_device.data_handle(),
           allowed_indices_host.data(),
           allowed_indices_host.size(),
           raft::resource::get_cuda_stream(res));

cuvs::core::bitset<uint32_t, indexing_dtype> allowed_indices_bitset(
    res,
    allowed_indices_device.view(),
    index.size());
auto bitset_filter =
    filtering::bitset_filter(allowed_indices_bitset.view());

ivf_flat::search(
    res,
    search_params,
    index,
    queries,
    neighbors.view(),
    distances.view(),
    bitset_filter);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp
import numpy as np
from cuvs.neighbors import filters, ivf_flat

n_vectors = dataset.shape[0]
allowed_indices = np.asarray(get_allowed_indices(), dtype=np.uint32)

bitset_host = np.zeros((n_vectors + 31) // 32, dtype=np.uint32)
for idx in allowed_indices:
    bitset_host[idx // 32] |= np.uint32(1 << (idx % 32))

prefilter = filters.from_bitset(cp.asarray(bitset_host))
search_params = ivf_flat.SearchParams(n_probes=20)

# ... build or load index ...
distances, neighbors = ivf_flat.search(
    search_params,
    index,
    queries,
    k,
    filter=prefilter,
)
```

</Tab>
</Tabs>

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / `sqeuclidean` | Distance metric used for cluster training and search. |
| `metric_arg` | `2.0` | Extra argument for metrics that need one, such as Minkowski distance. |
| `n_lists` | `1024` | Number of coarse clusters, or inverted lists. More lists make each list smaller, but may require more probes to maintain recall. |
| `kmeans_n_iters` | `20` | Maximum number of k-means iterations used to train the list centers. |
| `kmeans_trainset_fraction` | `0.5` | Fraction of the dataset used for k-means training. Reducing this can speed up build on very large datasets. |
| `adaptive_centers` | `False` | Updates centers when new vectors are added. This can help when appended data shifts distribution, but center values then depend on insertion order. |
| `conservative_memory_allocation` | `False` | Uses tighter per-list allocations. The default overallocates lists to reduce reallocations during repeated `extend` calls. |
| `add_data_on_build` | `True` | Adds vectors to the index during build. Set to `False` to train the centers first and add vectors later with `extend`. |

### Search parameters

| Name | Default | Description |
| --- | --- | --- |
| `n_probes` | `20` | Number of closest lists to scan for each query. This is the main recall and latency knob. |
| `metric_udf` | None | Optional custom metric UDF code used by the C++ search path. |

Filters are passed as search function arguments in bindings that expose filtered search, not as fields in `ivf_flat::search_params`.

## Tuning

Start with `n_lists`. A common starting point is near `sqrt(N)`, then adjust based on recall, latency, and list balance.

Tune `n_probes` next. Increasing `n_probes` usually improves recall because each query scans more lists, but it also does more distance computation.

If build time is high, reduce `kmeans_trainset_fraction` or `kmeans_n_iters`. If recall is poor even with higher `n_probes`, the trained clusters may be too coarse or poorly matched to the data.

For append-heavy workflows, decide whether `add_data_on_build`, `adaptive_centers`, and `conservative_memory_allocation` match the update pattern. Tighter allocation uses less memory, while the default allocation strategy can make repeated `extend` calls cheaper.

## Memory footprint

IVF-Flat memory has three main parts: full-precision vectors stored in inverted lists, source row IDs for those vectors, and cluster centers. The index does not need the original dataset after build because it stores the vectors in its own list layout.

To keep the formulas readable, this section uses short symbols. All estimates are in bytes. The examples convert bytes to MiB by dividing by `1024 * 1024`.

- `N`: Number of database vectors, or rows in the dataset being indexed.
- `D`: Vector dimension, or number of values in each vector.
- `B`: Bytes stored for each vector value. Use `4` for fp32, `2` for fp16, or `1` for int8 and uint8.
- `L`: Number of inverted lists. This is the `n_lists` build parameter.
- `N_cap`: Total allocated list capacity across all lists after padding and over-allocation.
- `S_idx`: Bytes per source row ID. This is `sizeof(IdxT)`, usually `8` for `int64_t`.
- `Q`: Query batch size, or number of query vectors processed together.
- `K`: Search result count, or the requested `k` nearest neighbors per query.
- `P`: Number of lists probed during search. This is the `n_probes` search parameter.
- `F`: Training-set fraction. This is the `kmeans_trainset_fraction` build parameter.

The named terms in the formulas are also memory sizes:

- `list_data_size`: Device memory used by vectors stored inside IVF lists.
- `list_index_size`: Device memory used by source row IDs stored beside list vectors.
- `centers_size`: Device memory used by the `L` cluster centers.
- `index_size`: Approximate device memory used by the trained IVF-Flat index.
- `query_size`: Device memory for the current query batch.
- `result_size`: Device memory for neighbor IDs and distances returned for the current query batch.
- `workspace_size`: Temporary search workspace.

### Scratch and maximum vectors

The formulas below show the major persistent and temporary buffers. Additional scratch comes from k-means workspaces, list allocation padding, allocator fragmentation, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.20` for build estimates and `H = 0.10` for search estimates. If you can measure a representative smaller run, use:

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


IVF-Flat uses padded list capacity, so estimate `N_cap` first. With balanced lists, substitute the approximation for `N_cap`; with measured list sizes, use the exact sum. Then solve the selected peak formula:

$$
\text{peak\_without\_scratch}(N)
  = N_{\text{cap}}(N) \cdot B_{\text{per\_stored\_vector}}
    + N \cdot B_{\text{per\_input\_vector}}
    + B_{\text{fixed}}
$$

Because `N_cap(N)` is stepwise, the most reliable capacity estimate is to try increasing `N` values until `peak_without_scratch(N) <= M_usable` no longer holds. For rough planning, use `N_cap ≈ N` plus 5-10% padding when lists are balanced, or more if many lists are small.

### List capacity

Each inverted list is padded for efficient memory access. The minimum alignment is 32 rows. With the default allocation strategy, lists may be overallocated up to 1024-row chunks to make future `extend` calls cheaper. With `conservative_memory_allocation = true`, the maximum alignment is 32 rows.

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

The implementation chooses a capacity for each list based on `A_min` and `A_max`:

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

Use the exact list sizes when estimating tightly. Empty or very small lists still carry padding overhead, so choosing far more lists than needed can waste memory.

### Baseline memory after build

The baseline memory footprint after index construction is:

$$
\begin{aligned}
\text{list\_data\_size}
&= N_{\text{cap}} \times D \times B
\end{aligned}
$$

$$
\begin{aligned}
\text{list\_index\_size}
&= N_{\text{cap}} \times S_{\text{idx}}
\end{aligned}
$$

$$
\begin{aligned}
\text{centers\_size}
&= L \times D
  \times \operatorname{sizeof}(\mathrm{float})
\end{aligned}
$$

$$
\begin{aligned}
\text{index\_size}
&\approx
  \text{list\_data\_size} \\
&\quad + \text{list\_index\_size} \\
&\quad + \text{centers\_size} \\
&\quad + \text{small list metadata}
\end{aligned}
$$

**Example** (`N = 1e6`, `D = 128`, `B = 4`, `L = 1024`, balanced lists, default allocation, `IdxT = int64`):

- `N / L = 976.56`, so `N_cap ~= 1024 * 1024 = 1,048,576`
- `list_data_size = 536,870,912 B = 512.00 MiB`
- `list_index_size = 8,388,608 B = 8.00 MiB`
- `centers_size = 524,288 B = 0.50 MiB`
- `index_size ~= 520.50 MiB + small metadata`

### Build peak memory usage

Build trains k-means centers, then fills the lists when `add_data_on_build = true`. Peak memory depends on whether the input dataset is already on device and how the workspace memory resource is configured.

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

$$
\begin{aligned}
\text{build\_peak}
&\approx
  \text{input\_dataset\_size} \\
&\quad + \text{training\_sample\_size} \\
&\quad + \text{training\_labels\_size} \\
&\quad + \text{centers\_size} \\
&\quad + \text{index\_size}
\end{aligned}
$$

When the input is already on device, `input_dataset_size = N * D * B`. Host inputs may stage batches on the device instead of adding the full dataset as a separate device allocation.

### Search peak memory usage

IVF-Flat search requires the index to be resident on the GPU. Search also needs query vectors, output buffers, and temporary workspace.

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

A useful workspace estimate is:

$$
\begin{aligned}
\text{workspace\_size}
&\lesssim
\min\!\Big(
  1\ \mathrm{GiB}, \\
&\qquad
  Q \times
  \Big[
    \big(L + 1 + P \times (K + 1)\big)
    \times \operatorname{sizeof}(\mathrm{float}) \\
&\qquad\qquad
    + P \times K \times S_{\text{idx}}
  \Big]
\Big)
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
