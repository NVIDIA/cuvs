# Brute-force

Brute-force, or a flat index, is the simplest nearest-neighbor index. It stores the dataset and checks every vector during search.

Brute-force works well when exact results matter, the dataset is small enough to scan, or filters remove most of the dataset before search.

## Example API Usage

[C API](/api-reference/c-api-neighbors-brute-force) | [C++ API](/api-reference/cpp-api-neighbors-brute-force) | [Python API](/api-reference/python-api-neighbors-brute-force) | [Java API](/api-reference/java-api-com-nvidia-cuvs-bruteforceindex) | [Rust API](/api-reference/rust-api-cuvs-brute-force) | [Go API](/api-reference/go-api-brute-force)

### Building an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/brute_force.h>

cuvsResources_t res;
cuvsBruteForceIndex_t index;
DLManagedTensor *dataset;

load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsBruteForceIndexCreate(&index);

cuvsBruteForceBuild(res, dataset, L2Expanded, 0.0f, index);

cuvsBruteForceIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/brute_force.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
brute_force::index_params index_params;

index_params.metric = cuvs::distance::DistanceType::L2Expanded;

auto index = brute_force::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import brute_force

dataset = load_data()

index = brute_force.build(dataset, metric="sqeuclidean")
```

</Tab>
<Tab title="Java">

```java
try (CuVSResources resources = CuVSResources.create();
    CuVSMatrix dataset = loadDatasetMatrix()) {
  BruteForceIndexParams indexParams =
      new BruteForceIndexParams.Builder()
          .withNumWriterThreads(32)
          .build();

  try (BruteForceIndex index =
      BruteForceIndex.newBuilder(resources)
          .withDataset(dataset)
          .withIndexParams(indexParams)
          .build()) {
    // Use index for search or serialization.
  }
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::brute_force::Index;
use cuvs::distance_type::DistanceType;
use cuvs::{Resources, Result};

fn build_brute_force_index(dataset: &ndarray::Array2<f32>) -> Result<Index> {
    let res = Resources::new()?;

    Index::build(&res, DistanceType::L2Expanded, None, dataset)
}
```

</Tab>
<Tab title="Go">

```go
package main

import (
	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/brute_force"
)

func buildBruteForceIndex(dataset cuvs.Tensor[float32]) (*brute_force.BruteForceIndex, error) {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return nil, err
	}

	index, err := brute_force.CreateIndex()
	if err != nil {
		return nil, err
	}

	_, err = dataset.ToDevice(&resource)
	if err != nil {
		return nil, err
	}

	err = brute_force.BuildIndex(resource, &dataset, cuvs.DistanceL2, 2.0, index)
	return index, err
}
```

</Tab>
</Tabs>

### Searching an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/brute_force.h>

cuvsResources_t res;
cuvsBruteForceIndex_t index;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

load_queries(queries);
allocate_outputs(neighbors, distances);

cuvsResourcesCreate(&res);

// ... build or load index ...
cuvsFilter prefilter = {0, NO_FILTER};
cuvsBruteForceSearch(res, index, queries, neighbors, distances, prefilter);

cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/brute_force.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
brute_force::index<float, float> index(res);
auto queries = load_queries();
auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
brute_force::search_params search_params;

// ... build or load index ...
brute_force::search(
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
from cuvs.neighbors import brute_force

queries = load_queries()
k = 10

# ... build or load index ...
distances, neighbors = brute_force.search(index, queries, k)
```

</Tab>
<Tab title="Java">

```java
BruteForceQuery query =
    new BruteForceQuery.Builder(resources)
        .withTopK(10)
        .withQueryVectors(queries)
        .build();

// ... build or load index ...
SearchResults results = index.search(query);
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::{ManagedTensor, Resources, Result};

fn search_brute_force_index(
    res: &Resources,
    index: &cuvs::brute_force::Index,
    queries: &ndarray::ArrayView2<f32>,
    k: usize,
) -> Result<()> {
    let n_queries = queries.shape()[0];
    let queries = ManagedTensor::from(queries).to_device(res)?;

    let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(res)?;

    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(res)?;

    index.search(res, &queries, &neighbors, &distances)?;

    distances.to_host(res, &mut distances_host)?;
    neighbors.to_host(res, &mut neighbors_host)?;

    Ok(())
}
```

</Tab>
<Tab title="Go">

```go
_, err := queries.ToDevice(&resource)
if err != nil {
	return err
}

err = brute_force.SearchIndex(resource, *index, &queries, &neighbors, &distances)
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

Serialize a Brute-force index when you want to reuse the same stored dataset without rebuilding it. The examples below save the dataset with the index so the loaded index can be searched immediately.

Rust and Go do not currently expose Brute-force save/load wrappers.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/brute_force.h>

cuvsResources_t res;
cuvsBruteForceIndex_t index;
cuvsBruteForceIndex_t loaded_index;

cuvsResourcesCreate(&res);
cuvsBruteForceIndexCreate(&index);
cuvsBruteForceIndexCreate(&loaded_index);

// ... build index ...
cuvsBruteForceSerialize(res, "/tmp/cuvs-bruteforce.bin", index);
cuvsBruteForceDeserialize(res, "/tmp/cuvs-bruteforce.bin", loaded_index);

cuvsBruteForceIndexDestroy(loaded_index);
cuvsBruteForceIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/brute_force.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
auto index = brute_force::build(res, brute_force::index_params{}, dataset);

brute_force::serialize(res, "/tmp/cuvs-bruteforce.bin", index, true);

brute_force::index<float, float> loaded_index(res);
brute_force::deserialize(res, "/tmp/cuvs-bruteforce.bin", &loaded_index);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import brute_force

dataset = load_data()
index = brute_force.build(dataset, metric="sqeuclidean")

brute_force.save("/tmp/cuvs-bruteforce.bin", index, include_dataset=True)
loaded_index = brute_force.load("/tmp/cuvs-bruteforce.bin")
```

</Tab>
<Tab title="Java">

```java
try (CuVSResources resources = CuVSResources.create();
    CuVSMatrix dataset = loadDatasetMatrix();
    BruteForceIndex index =
      BruteForceIndex.newBuilder(resources)
          .withDataset(dataset)
          .build()) {

  try (FileOutputStream output = new FileOutputStream("/tmp/cuvs-bruteforce.bin")) {
    index.serialize(output);
  }

  try (FileInputStream input = new FileInputStream("/tmp/cuvs-bruteforce.bin")) {
    BruteForceIndex loadedIndex =
        BruteForceIndex.newBuilder(resources)
            .from(input)
            .build();
  }
}
```

</Tab>
</Tabs>

## How Brute-force works

Brute-force stores the vectors and compares each query against every vector in the dataset. There is no partitioning step, graph traversal, or approximation step.

The build step is mostly a data setup step. The search step does the real work: for each query, NVIDIA cuVS computes distances to every vector, keeps the best `k` candidates, and returns exact nearest neighbors.

### Stream pools

The brute-force implementation can use the stream pool attached to the underlying resources object when it has independent chunks of work to schedule. A stream pool gives NVIDIA cuVS several CUDA streams to work with, which can improve GPU utilization for larger searches or partitioned inputs.

Stream pools are configured on the resources object, not through `brute_force::search_params`. Most users should start with the default resources behavior and add a small stream pool only after benchmarking shows a benefit. See [Stream pools](/user-guide/api-guides/core-types/resources#stream-pools) in the Resources guide for setup details.

## When to use Brute-force

Use brute-force when exact nearest neighbors are required.

Use brute-force when the dataset fits comfortably in device memory and search latency is acceptable.

Use brute-force as a ground-truth baseline for tuning approximate indexes.

Use brute-force for heavily filtered queries. If a filter excludes most vectors, brute-force can search only the remaining candidates while IVF and graph indexes may skip useful vectors before the filter is applied.

Use an approximate index instead when the dataset is large enough that scanning every vector is too slow.

## Using Filters

Brute-force supports filtered search. A filter tells the search which vectors are allowed before distances are computed.

This is useful when queries should only search a small subset of the dataset. For example, if a filter removes 90%-99% of the vectors, brute-force may do much less work while still returning exact results within the allowed subset.

Unlike partitioned or graph-based indexes, brute-force does not need to guess which part of the index might contain the answer. If a vector passes the filter, it can be considered.

The examples below use a bitmap filter. A bitmap can express a different allow list for each query. A bit value of `1` means the query may consider that vector; a bit value of `0` means the vector is filtered out for that query.

These examples cover the Brute-force bindings that currently expose filters. The Rust and Go Brute-force wrappers currently search without a filter argument.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/brute_force.h>

cuvsResources_t res;
cuvsBruteForceIndex_t index;
DLManagedTensor *dataset;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

cuvsResourcesCreate(&res);
cuvsBruteForceIndexCreate(&index);

load_dataset(dataset);
load_queries(queries);
allocate_outputs(neighbors, distances);

cuvsBruteForceBuild(res, dataset, L2Expanded, 0.0f, index);

// Create a device uint32 bitmap with one bit per (query, vector) pair.
// Bit 1 means the pair is allowed; bit 0 means it is filtered out.
DLManagedTensor *bitmap = make_device_bitmap(allowed_query_vector_pairs, n_queries, n_vectors);

cuvsFilter prefilter;
prefilter.type = BITMAP;
prefilter.addr = (uintptr_t)bitmap;

cuvsBruteForceSearch(res, index, queries, neighbors, distances, prefilter);

cuvsBruteForceIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/brute_force.hpp>

using namespace cuvs::neighbors;
using indexing_dtype = int64_t;

raft::device_resources res;
brute_force::index_params index_params;
brute_force::search_params search_params;
auto dataset = load_dataset(n_vectors, dim);
auto queries = load_queries(n_queries, dim);

auto index = brute_force::build(res, index_params, dataset);

// Store allowed pairs as flattened row-major ids:
// flattened_id = query_id * n_vectors + vector_id.
std::vector<indexing_dtype> allowed_pairs_host = get_allowed_query_vector_pairs();
auto allowed_pairs_device =
    raft::make_device_vector<indexing_dtype, indexing_dtype>(res, allowed_pairs_host.size());

raft::copy(allowed_pairs_device.data_handle(),
           allowed_pairs_host.data(),
           allowed_pairs_host.size(),
           raft::resource::get_cuda_stream(res));

cuvs::core::bitset<uint32_t, indexing_dtype> allowed_pairs_bitset(
    res, allowed_pairs_device.view(), n_queries * n_vectors);
cuvs::core::bitmap_view<const uint32_t, indexing_dtype> allowed_pairs_bitmap(
    allowed_pairs_bitset.data(), n_queries, n_vectors);

auto bitmap_filter = filtering::bitmap_filter(allowed_pairs_bitmap);

auto neighbors = raft::make_device_matrix<int64_t, indexing_dtype>(res, n_queries, k);
auto distances = raft::make_device_matrix<float, indexing_dtype>(res, n_queries, k);

brute_force::search(
    res,
    search_params,
    index,
    queries,
    neighbors.view(),
    distances.view(),
    bitmap_filter);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp
import numpy as np
from cuvs.neighbors import brute_force, filters

n_queries = queries.shape[0]
n_vectors = dataset.shape[0]
k = 10

index = brute_force.build(dataset, metric="sqeuclidean")

bitmap_host = np.zeros((n_queries * n_vectors + 31) // 32, dtype=np.uint32)
for query_id, vector_id in get_allowed_query_vector_pairs():
    bit = query_id * n_vectors + vector_id
    bitmap_host[bit // 32] |= np.uint32(1 << (bit % 32))

prefilter = filters.from_bitmap(cp.asarray(bitmap_host))

distances, neighbors = brute_force.search(
    index,
    queries,
    k,
    prefilter=prefilter,
)
```

</Tab>
<Tab title="Java">

```java
import java.util.BitSet;

BitSet[] prefilters = new BitSet[queries.length];
for (int queryId = 0; queryId < queries.length; queryId++) {
  BitSet allowed = new BitSet(vectors.length);
  for (int vectorId : getAllowedRowsForQuery(queryId)) {
    allowed.set(vectorId);
  }
  prefilters[queryId] = allowed;
}

BruteForceQuery query =
    new BruteForceQuery.Builder(resources)
        .withTopK(10)
        .withQueryVectors(queries)
        .withPrefilters(prefilters, vectors.length)
        .build();

// ... build or load index ...
SearchResults results = index.search(query);
```

</Tab>
</Tabs>

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / `sqeuclidean` | Distance metric used for search. |
| `metric_arg` | `2.0` | Extra argument for metrics that need one, such as Minkowski distance. |

### Search parameters

The C++ `brute_force::search_params` struct currently has no tunable fields.

Filters are passed as search function arguments in bindings that expose filtered search, not as fields in `brute_force::search_params`.

## Tuning Considerations

Brute-force has very few tuning knobs. The main decisions are distance metric, query batch size, output `k`, and whether a filter can reduce the amount of work.

Brute-force is exact, but ties can make result order look different across runs. If many vectors have the same distance near the cutoff, increasing `k` can make comparisons against ground truth more stable.

For large batches, memory allocation can affect performance. Reuse output buffers and memory resources when the API supports it.

## Memory footprint

Brute-force memory is dominated by the dataset, optional vector norms, current query batch, and output buffers. It does not build an additional graph or partitioned index.

Variables:

- `N`: Number of database vectors.
- `D`: Vector dimension.
- `B`: Bytes per dataset value, such as `4` for fp32.
- `Q`: Query batch size.
- `K`: Number of neighbors returned per query.
- `S_idx`: Bytes per returned neighbor ID.
- `R_norm`: `1` when the selected distance metric stores vector norms, otherwise `0`.

### Scratch and maximum vectors

The `scratch` term covers temporary distance tiles, allocator padding, CUDA library workspaces, and memory held by the active memory resource. Use `H = 0.10` for a first search estimate. If you can measure a representative run, use:

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


For fixed `D`, `Q`, and `K`, solve:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_vector}}}
  \right\rfloor
$$

where `B_per_vector = D * B + R_norm * 4`. `B_fixed` includes query and result buffers.

### Index footprint

The raw vectors use:

$$
\text{dataset\_size}
  = N \cdot D \cdot B
$$

Vector norms, for distance metrics that use them, use:

$$
\text{norms\_size}
  = R_{\text{norm}} \cdot N \cdot 4
$$

### Search peak memory usage

Search adds the query batch and output buffers:

$$
\begin{aligned}
\text{query\_size} &= Q \cdot D \cdot B \\
\text{result\_size} &= Q \cdot K \cdot (S_{\text{idx}} + 4)
\end{aligned}
$$

The total search peak is approximately:

$$
\begin{aligned}
\text{search\_peak}
  \approx&\ \text{dataset\_size}
   + \text{norms\_size} \\
  &+ \text{query\_size}
   + \text{result\_size}
   + \text{scratch}
\end{aligned}
$$
