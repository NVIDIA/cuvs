# CAGRA

CAGRA is a GPU-optimized graph index for approximate nearest-neighbor search. Think of every vector as a point, and think of the graph as a map that connects each point to nearby points. During search, CAGRA follows that map toward better matches instead of checking every vector.

CAGRA works well when you want strong recall, high GPU throughput, and fast graph construction.

## Example API Usage

[C API](/api-reference/c-api-neighbors-cagra) | [C++ API](/api-reference/cpp-api-neighbors-cagra) | [Python API](/api-reference/python-api-neighbors-cagra) | [Java API](/api-reference/java-api-com-nvidia-cuvs-cagraindex) | [Rust API](/api-reference/rust-api-cuvs-cagra) | [Go API](/api-reference/go-api-cagra)

### Building an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/cagra.h>

cuvsResources_t res;
cuvsCagraIndexParams_t index_params;
cuvsCagraIndex_t index;
DLManagedTensor *dataset;

// populate tensor with data
load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsCagraIndexParamsCreate(&index_params);
cuvsCagraIndexCreate(&index);

cuvsCagraBuild(res, index_params, dataset, index);

cuvsCagraIndexDestroy(index);
cuvsCagraIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
raft::device_matrix_view<float> dataset = load_dataset();
cagra::index_params index_params;

auto index = cagra::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import cagra

dataset = load_data()
index_params = cagra.IndexParams()

index = cagra.build(index_params, dataset)
```

</Tab>
<Tab title="Java">

```java
try (CuVSResources resources = CuVSResources.create();
    CuVSMatrix dataset = loadDatasetMatrix()) {
  CagraIndexParams indexParams =
      new CagraIndexParams.Builder()
          .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
          .withMetric(CuvsDistanceType.L2Expanded)
          .build();

  try (CagraIndex index =
      CagraIndex.newBuilder(resources)
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
use cuvs::cagra::{Index, IndexParams};
use cuvs::{Resources, Result};

fn build_cagra_index(dataset: &ndarray::Array2<f32>) -> Result<Index> {
    let res = Resources::new()?;
    let index_params = IndexParams::new()?;

    Index::build(&res, &index_params, dataset)
}
```

</Tab>
<Tab title="Go">

```go
package main

import (
	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/cagra"
)

func buildCagraIndex(dataset cuvs.Tensor[float32]) (*cagra.CagraIndex, error) {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return nil, err
	}

	indexParams, err := cagra.CreateIndexParams()
	if err != nil {
		return nil, err
	}

	index, err := cagra.CreateIndex()
	if err != nil {
		return nil, err
	}

	_, err = dataset.ToDevice(&resource)
	if err != nil {
		return nil, err
	}

	err = cagra.BuildIndex(resource, indexParams, &dataset, index)
	return index, err
}
```

</Tab>
</Tabs>

### Extending an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/cagra.h>

cuvsResources_t res;
cuvsCagraExtendParams_t extend_params;
cuvsCagraIndex_t index;
DLManagedTensor *additional_dataset;

load_additional_dataset(additional_dataset);

cuvsResourcesCreate(&res);
cuvsCagraExtendParamsCreate(&extend_params);
cuvsCagraIndexCreate(&index);

// ... build or load index ...
cuvsCagraExtend(res, extend_params, additional_dataset, index);

cuvsCagraIndexDestroy(index);
cuvsCagraExtendParamsDestroy(extend_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
cagra::index_params index_params;
cagra::extend_params extend_params;
auto dataset = load_dataset();
auto additional_dataset = load_additional_dataset();

auto index = cagra::build(res, index_params, dataset);
cagra::extend(res, extend_params, additional_dataset, index);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import cagra

dataset = load_data()
additional_dataset = load_additional_data()

index = cagra.build(cagra.IndexParams(), dataset)
index = cagra.extend(cagra.ExtendParams(), index, additional_dataset)
```

</Tab>
<Tab title="Go">

```go
package main

import (
	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/cagra"
)

func extendCagraIndex(
	resource cuvs.Resource,
	index *cagra.CagraIndex,
	additionalDataset cuvs.Tensor[float32],
) error {
	extendParams, err := cagra.CreateExtendParams()
	if err != nil {
		return err
	}
	defer extendParams.Close()

	_, err = additionalDataset.ToDevice(&resource)
	if err != nil {
		return err
	}

	return cagra.ExtendIndex(resource, extendParams, &additionalDataset, index)
}
```

</Tab>
</Tabs>

See the [C](/api-reference/c-api-neighbors-cagra#cuvscagraextend), [C++](/api-reference/cpp-api-neighbors-cagra#neighbors-cagra-extend), [Python](/api-reference/python-api-neighbors-cagra#extend), and [Go](/api-reference/go-api-cagra#extendindex) API references for the full signatures.

### Searching an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/cagra.h>

cuvsResources_t res;
cuvsCagraSearchParams_t search_params;
cuvsCagraIndex_t index;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

// populate tensor with data
load_queries(queries);

cuvsResourcesCreate(&res);
cuvsCagraSearchParamsCreate(&search_params);

// ... build or load index ...
cuvsCagraSearch(res, search_params, index, queries, neighbors, distances);

cuvsCagraSearchParamsDestroy(search_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
cagra::index index;
raft::device_matrix_view<float> queries = load_queries();
raft::device_matrix_view<uint32_t> neighbors = make_device_matrix_view<uint32_t>(n_queries, k);
raft::device_matrix_view<float> distances = make_device_matrix_view<float>(n_queries, k);
cagra::search_params search_params;

// ... build or load index ...
cagra::search(res, search_params, index, queries, neighbors, distances);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import cagra

queries = load_queries()
search_params = cagra.SearchParams()

# ... build or load index ...
neighbors, distances = cagra.search(search_params, index, queries, k)
```

</Tab>
<Tab title="Java">

```java
CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();

try (CuVSMatrix queryVectors = loadQueryMatrix()) {
  CagraQuery cuvsQuery =
    new CagraQuery.Builder()
        .withTopK(10)
        .withSearchParams(searchParams)
        .withQueryVectors(queryVectors)
        .build();

  // ... build or load index ...
  SearchResults results = index.search(cuvsQuery);
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::cagra::SearchParams;
use cuvs::{ManagedTensor, Resources, Result};

fn search_cagra_index(
    res: &Resources,
    index: &cuvs::cagra::Index,
    queries: &ndarray::ArrayView2<f32>,
    k: usize,
) -> Result<()> {
    let n_queries = queries.shape()[0];
    let queries = ManagedTensor::from(queries).to_device(res)?;

    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from(&neighbors_host).to_device(res)?;

    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from(&distances_host).to_device(res)?;

    let search_params = SearchParams::new()?;
    index.search(res, &search_params, &queries, &neighbors, &distances)?;

    distances.to_host(res, &mut distances_host)?;
    neighbors.to_host(res, &mut neighbors_host)?;

    Ok(())
}
```

</Tab>
<Tab title="Go">

```go
searchParams, err := cagra.CreateSearchParams()
if err != nil {
	return err
}
defer searchParams.Close()

_, err = queries.ToDevice(&resource)
if err != nil {
	return err
}

err = cagra.SearchIndex(resource, searchParams, index, &queries, &neighbors, &distances, nil)
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

Serialize a CAGRA index when you want to reuse the graph without rebuilding it. Include the dataset when the loaded index should be searchable immediately; omit it only when your workflow will attach or provide the dataset separately.

Go does not currently expose CAGRA save/load wrappers.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/cagra.h>

cuvsResources_t res;
cuvsCagraIndex_t index;
cuvsCagraIndex_t loaded_index;

cuvsResourcesCreate(&res);
cuvsCagraIndexCreate(&index);
cuvsCagraIndexCreate(&loaded_index);

// ... build index ...
cuvsCagraSerialize(res, "/tmp/cuvs-cagra.bin", index, true);
cuvsCagraDeserialize(res, "/tmp/cuvs-cagra.bin", loaded_index);

cuvsCagraIndexDestroy(loaded_index);
cuvsCagraIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
auto index = cagra::build(res, cagra::index_params{}, dataset);

cagra::serialize(res, "/tmp/cuvs-cagra.bin", index, true);

cagra::index<float, uint32_t> loaded_index(res);
cagra::deserialize(res, "/tmp/cuvs-cagra.bin", &loaded_index);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import cagra

dataset = load_data()
index = cagra.build(cagra.IndexParams(), dataset)

cagra.save("/tmp/cuvs-cagra.bin", index, include_dataset=True)
loaded_index = cagra.load("/tmp/cuvs-cagra.bin")
```

</Tab>
<Tab title="Java">

```java
try (CuVSResources resources = CuVSResources.create();
    CuVSMatrix dataset = loadDatasetMatrix();
    CagraIndex index =
      CagraIndex.newBuilder(resources)
          .withDataset(dataset)
          .build()) {

  try (FileOutputStream output = new FileOutputStream("/tmp/cuvs-cagra.bin")) {
    index.serialize(output);
  }

  try (FileInputStream input = new FileInputStream("/tmp/cuvs-cagra.bin")) {
    CagraIndex loadedIndex =
        CagraIndex.newBuilder(resources)
            .from(input)
            .build();
  }
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::cagra::{Index, IndexParams};
use cuvs::{Resources, Result};

fn save_and_load_cagra(dataset: &ndarray::Array2<f32>) -> Result<Index> {
    let res = Resources::new()?;
    let index_params = IndexParams::new()?;
    let index = Index::build(&res, &index_params, dataset)?;

    index.serialize(&res, "/tmp/cuvs-cagra.bin", true)?;
    Index::deserialize(&res, "/tmp/cuvs-cagra.bin")
}
```

</Tab>
</Tabs>

## How CAGRA works

CAGRA builds and searches a nearest-neighbor graph.

First, CAGRA builds an initial kNN graph. This is the first draft of the map: each vector is connected to vectors that look nearby. An exact brute-force build can create a very accurate initial graph, but it is usually too slow. In practice, the first graph does not need to be perfect because CAGRA improves it later. NVIDIA cuVS can build this initial graph with IVF-PQ or NN-Descent.

Second, CAGRA prunes the initial graph. This removes redundant paths and keeps the links that are most useful for search.

At search time, CAGRA starts from one or more graph vertices, follows links to better candidates, and keeps a working set of the best candidates it has seen so far.

## When to use CAGRA

Use CAGRA when the index fits in GPU memory and you want fast approximate search.

Use CAGRA when build speed matters. CAGRA can build graphs quickly on the GPU.

Use CAGRA in hybrid environments where a GPU-built graph is converted to HNSW for CPU search.

Use brute-force instead when exact results are required or the dataset is small enough that a full scan is already fast enough.

## Interoperability with HNSW

NVIDIA cuVS can convert a CAGRA graph to an HNSW graph. This lets the GPU build the graph while the CPU handles search later. This is useful when GPUs are available for indexing, but production search runs on CPUs.

If the graph is being serialized or converted to HNSW right after build, avoid keeping the dataset attached to the CAGRA index when the binding exposes that option. In C++, for example, set `attach_dataset_on_build` to `false`.

These examples cover the bindings that currently expose CAGRA and HNSW interoperability. Go supports CAGRA build and search, but does not currently expose HNSW conversion.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>

cuvsResources_t res;
cuvsCagraIndexParams_t cagra_params;
cuvsCagraIndex_t cagra_index;
cuvsHnswIndexParams_t hnsw_params;
cuvsHnswIndex_t hnsw_index;
cuvsHnswSearchParams_t hnsw_search_params;
DLManagedTensor *dataset;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

int64_t n_rows = 1000000;
int64_t dim = 128;
int M = 32;
int ef_construction = 200;

cuvsResourcesCreate(&res);
cuvsCagraIndexParamsCreate(&cagra_params);
cuvsCagraIndexCreate(&cagra_index);
cuvsHnswIndexParamsCreate(&hnsw_params);
cuvsHnswIndexCreate(&hnsw_index);
cuvsHnswSearchParamsCreate(&hnsw_search_params);

load_dataset(dataset);
load_host_queries(queries);
allocate_hnsw_outputs(neighbors, distances);

cuvsCagraIndexParamsFromHnswParams(
    cagra_params,
    n_rows,
    dim,
    M,
    ef_construction,
    CUVS_CAGRA_HEURISTIC_SIMILAR_SEARCH_PERFORMANCE,
    L2Expanded);

hnsw_params->hierarchy = GPU;
hnsw_search_params->ef = 200;
hnsw_search_params->num_threads = 0;

cuvsCagraBuild(res, cagra_params, dataset, cagra_index);
cuvsHnswFromCagra(res, hnsw_params, cagra_index, hnsw_index);
cuvsHnswSearch(res, hnsw_search_params, hnsw_index, queries, neighbors, distances);

cuvsHnswSearchParamsDestroy(hnsw_search_params);
cuvsHnswIndexDestroy(hnsw_index);
cuvsHnswIndexParamsDestroy(hnsw_params);
cuvsCagraIndexDestroy(cagra_index);
cuvsCagraIndexParamsDestroy(cagra_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>

using namespace cuvs::neighbors;

raft::resources res;
auto dataset = load_device_dataset();
auto queries = load_host_queries();

constexpr int M = 32;
constexpr int ef_construction = 200;
constexpr int64_t n_queries = 1000;
constexpr int64_t k = 10;

auto cagra_params =
    cagra::index_params::from_hnsw_params(dataset.extents(), M, ef_construction);
cagra_params.attach_dataset_on_build = false;

auto cagra_index = cagra::build(res, cagra_params, dataset);

hnsw::index_params hnsw_params;
hnsw_params.hierarchy = hnsw::HnswHierarchy::GPU;

auto hnsw_index = hnsw::from_cagra(res, hnsw_params, cagra_index);

hnsw::search_params search_params;
search_params.ef = 200;
search_params.num_threads = 0;

auto neighbors = raft::make_host_matrix<uint64_t, int64_t>(res, n_queries, k);
auto distances = raft::make_host_matrix<float, int64_t>(res, n_queries, k);

hnsw::search(
    res,
    search_params,
    *hnsw_index,
    queries,
    neighbors.view(),
    distances.view());
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import cagra, hnsw

dataset = load_data()
queries = load_host_queries()
k = 10

cagra_index = cagra.build(cagra.IndexParams(), dataset)

hnsw_params = hnsw.IndexParams(hierarchy="gpu")
hnsw_index = hnsw.from_cagra(hnsw_params, cagra_index)

search_params = hnsw.SearchParams(ef=200, num_threads=0)
distances, neighbors = hnsw.search(search_params, hnsw_index, queries, k)
```

</Tab>
<Tab title="Java">

```java
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.HnswIndexParams.HnswHierarchy;
import com.nvidia.cuvs.HnswQuery;
import com.nvidia.cuvs.HnswSearchParams;
import com.nvidia.cuvs.SearchResults;

try (CuVSResources resources = CuVSResources.create();
    CuVSMatrix dataset = loadDatasetMatrix()) {
  int dim = (int) dataset.columns();
  int M = 32;
  int efConstruction = 200;

  CagraIndexParams cagraParams =
      CagraIndexParams.fromHnswParams(
          (int) dataset.size(),
          dim,
          M,
          efConstruction,
          HnswHeuristicType.SIMILAR_SEARCH_PERFORMANCE,
          CuvsDistanceType.L2Expanded);

  try (CagraIndex cagraIndex =
      CagraIndex.newBuilder(resources)
          .withDataset(dataset)
          .withIndexParams(cagraParams)
          .build()) {
    HnswIndexParams hnswParams =
        new HnswIndexParams.Builder()
            .withVectorDimension(dim)
            .withHierarchy(HnswHierarchy.GPU)
            .build();

    try (HnswIndex hnswIndex = HnswIndex.fromCagra(hnswParams, cagraIndex)) {
      HnswSearchParams searchParams =
          new HnswSearchParams.Builder()
              .withEF(200)
              .withNumThreads(0)
              .build();
      HnswQuery query =
          new HnswQuery.Builder(resources)
              .withTopK(10)
              .withSearchParams(searchParams)
              .withQueryVectors(queries)
              .build();

      SearchResults results = hnswIndex.search(query);
    }
  }
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::cagra::{Index, IndexParams};
use cuvs::{Resources, Result};

fn export_cagra_as_hnswlib(dataset: &ndarray::Array2<f32>) -> Result<()> {
    let res = Resources::new()?;
    let index_params = IndexParams::new()?;
    let index = Index::build(&res, &index_params, dataset)?;

    index.serialize_to_hnswlib(&res, "cagra_graph.hnsw")
}
```

</Tab>
</Tabs>

## Using Filters

CAGRA supports filtered search. A filter hides some vectors from the search result, so CAGRA may need to explore more of the graph to find enough valid neighbors.

CAGRA can adjust `itopk_size` internally based on the filtering rate. To disable this automatic adjustment, set `filtering_rate` to `0.0`.

The examples below use a bitset filter. A bit value of `1` means a vector is allowed; a bit value of `0` means it is filtered out.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>

cuvsResources_t res;
cuvsCagraIndexParams_t index_params;
cuvsCagraSearchParams_t search_params;
cuvsCagraIndex_t index;
DLManagedTensor *dataset;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

cuvsResourcesCreate(&res);
cuvsCagraIndexParamsCreate(&index_params);
cuvsCagraSearchParamsCreate(&search_params);
cuvsCagraIndexCreate(&index);

// Populate DLPack tensors with dataset, query, and output data.
load_dataset(dataset);
load_queries(queries);
allocate_outputs(neighbors, distances);

cuvsCagraBuild(res, index_params, dataset, index);

// Create a device uint32 bitset with one bit per indexed vector. Bit 1 means
// allowed; bit 0 means filtered out.
DLManagedTensor *bitset = make_device_bitset(allowed_indices, n_vectors);

cuvsFilter filter;
filter.type = BITSET;
filter.addr = (uintptr_t)bitset;

cuvsCagraSearch(res, search_params, index, queries, neighbors, distances, filter);

cuvsCagraIndexDestroy(index);
cuvsCagraSearchParamsDestroy(search_params);
cuvsCagraIndexParamsDestroy(index_params);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/core/bitset.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
cagra::index index;
cagra::search_params search_params;
raft::device_matrix_view<float> queries = load_queries();
raft::device_matrix_view<uint32_t> neighbors = make_device_matrix_view<uint32_t>(n_queries, k);
raft::device_matrix_view<float> distances = make_device_matrix_view<float>(n_queries, k);

// Load a list of all samples that should be filtered out.
std::vector<uint32_t> removed_indices_host = get_invalid_indices();
auto removed_indices_device =
      raft::make_device_vector<uint32_t, uint32_t>(res, removed_indices_host.size());

raft::copy(removed_indices_device.data_handle(),
           removed_indices_host.data(),
           removed_indices_host.size(),
           raft::resource::get_cuda_stream(res));

// Create a bitset and pass it to CAGRA search.
cuvs::core::bitset<uint32_t, uint32_t> removed_indices_bitset(
    res, removed_indices_device.view(), index.size());
auto bitset_filter =
      cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());

cagra::search(res, search_params, index, queries, neighbors, distances, bitset_filter);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp
import numpy as np
from cuvs.neighbors import cagra, filters

n_vectors = dataset.shape[0]
allowed_indices = np.asarray(get_allowed_indices(), dtype=np.uint32)

bitset_host = np.zeros((n_vectors + 31) // 32, dtype=np.uint32)
for idx in allowed_indices:
    bitset_host[idx // 32] |= np.uint32(1 << (idx % 32))

prefilter = filters.from_bitset(cp.asarray(bitset_host))
search_params = cagra.SearchParams()

# ... build or load index ...
distances, neighbors = cagra.search(
    search_params,
    index,
    queries,
    k,
    filter=prefilter,
)
```

</Tab>
<Tab title="Java">

```java
import java.util.BitSet;

int numDocs = (int) dataset.size();
BitSet prefilter = new BitSet(numDocs);
prefilter.set(0, numDocs);

for (int row : getFilteredRows()) {
  prefilter.clear(row);
}

CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();
try (CuVSMatrix queryVectors = loadQueryMatrix()) {
  CagraQuery cuvsQuery =
    new CagraQuery.Builder()
        .withTopK(10)
        .withSearchParams(searchParams)
        .withQueryVectors(queryVectors)
        .withPrefilter(prefilter, numDocs)
        .build();

  // ... build or load index ...
  SearchResults results = index.search(cuvsQuery);
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::cagra::SearchParams;
use cuvs::{ManagedTensor, Resources, Result};

fn search_with_filter(
    res: &Resources,
    index: &cuvs::cagra::Index,
    queries: &ManagedTensor,
    neighbors: &ManagedTensor,
    distances: &ManagedTensor,
    n_vectors: usize,
) -> Result<()> {
    let n_words = (n_vectors + 31) / 32;
    let mut bitset_host = ndarray::Array::<u32, _>::zeros(ndarray::Ix1(n_words));

    for idx in get_allowed_indices() {
        bitset_host[idx / 32] |= 1u32 << (idx % 32);
    }

    let bitset = ManagedTensor::from(&bitset_host).to_device(res)?;
    let search_params = SearchParams::new()?;

    index.search_with_filter(res, &search_params, queries, neighbors, distances, &bitset)
}
```

</Tab>
<Tab title="Go">

```go
searchParams, err := cagra.CreateSearchParams()
if err != nil {
	return err
}
defer searchParams.Close()

// The Go CAGRA API accepts an allow list and converts it to a bitset.
allowList := []uint32{0, 2, 4, 6}

err = cagra.SearchIndex(
	resource,
	searchParams,
	index,
	&queries,
	&neighbors,
	&distances,
	allowList,
)
if err != nil {
	return err
}
```

</Tab>
</Tabs>

### C++ filter UDFs

CAGRA C++ search can also use a JIT-LTO filter UDF when the predicate needs device metadata such as per-row tenants, ACLs, timestamps, or query-specific attributes. The UDF is a candidate-validity predicate: it decides whether a logical `source_id` is allowed for a logical `query_id`. It cannot control graph traversal, access PQ/VPQ or graph internals, or change distance computation.

```cpp
struct tenant_filter_context {
  const uint32_t* row_tenants;
  const uint32_t* query_tenants;
};

std::string source = R"cpp(
struct tenant_filter_context {
  const uint32_t* row_tenants;
  const uint32_t* query_tenants;
};

__device__ bool cuvs_filter_udf(uint32_t query_id,
                                source_index_t source_id,
                                void* filter_data)
{
  auto* ctx = static_cast<const tenant_filter_context*>(filter_data);
  return ctx->row_tenants[source_id] == ctx->query_tenants[query_id];
}
)cpp";

tenant_filter_context host_ctx{row_tenants_device, query_tenants_device};
tenant_filter_context* ctx_device = copy_to_device(host_ctx);

auto filter = cuvs::neighbors::filtering::udf_filter(source, ctx_device, 0.75f);

cagra::search(res, search_params, index, queries, neighbors, distances, filter);
```

The `filter_data` pointer is passed through unchanged to the device predicate. If the predicate dereferences it, the pointer and any nested pointers must refer to device-accessible memory and remain valid for the duration of the search. The `query_id` passed to the UDF is the global logical query id, including the batch offset when `max_queries` causes CAGRA to split a search into multiple batches.

If `search_params::filtering_rate` is negative, CAGRA uses `udf_filter::filtering_rate`. If both are negative, CAGRA assumes `0.0` because UDF selectivity cannot be inferred from arbitrary CUDA source. Because CAGRA remains approximate, filtered results are not guaranteed to match exact brute-force filtered search, especially for highly selective predicates without an accurate `filtering_rate`.

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / `sqeuclidean` | Distance metric used to build and search the graph. |
| `metric_arg` | `2.0` | Extra argument for metrics that need one, such as Minkowski distance. |
| `intermediate_graph_degree` | `128` | Number of neighbors kept in the initial graph before pruning. Larger values can improve the final graph, but increase build time and memory use. |
| `graph_degree` | `64` | Number of neighbors kept for each vertex in the final graph. Larger values can improve recall, but use more memory and search work. |
| `compression` | None | Optional vector product quantization parameters. When set, the compressed dataset is attached to the index and `attach_dataset_on_build` is effectively enabled. |
| `graph_build_params` | `std::monostate` | Parameters for the initial graph builder. The default lets NVIDIA cuVS choose a heuristic; explicit options include IVF-PQ, NN-Descent, ACE, and iterative-search graph build parameters. |
| `guarantee_connectivity` | `False` | Uses a degree-constrained minimum spanning tree to guarantee the initial kNN graph is connected. This can improve recall on some datasets. |
| `attach_dataset_on_build` | `True` | Keeps the dataset attached to the index after build. Set to `False` when serializing or converting to another graph format right after build. |

### Search parameters

| Name | Default | Description |
| --- | --- | --- |
| `max_queries` | `0` | Maximum number of queries searched concurrently. `0` lets NVIDIA cuVS choose automatically. |
| `itopk_size` | 64 | Number of intermediate search results kept during search. This must be at least `k` and is the main search tuning knob. |
| `max_iterations` | 0 | Maximum number of search iterations. `0` lets NVIDIA cuVS choose automatically. |
| `algo` | `AUTO` | Search implementation. Options include `SINGLE_CTA`, `MULTI_CTA`, `MULTI_KERNEL`, and `AUTO`. |
| `team_size` | 0 | Number of CUDA threads used to calculate each distance. Valid values are 4, 8, 16, or 32. `0` lets NVIDIA cuVS choose automatically. |
| `search_width` | 1 | Number of vertices selected as starting points for each search iteration. |
| `min_iterations` | 0 | Minimum number of search iterations. |
| `thread_block_size` | `0` | CUDA thread block size. Supported values include 64, 128, 256, 512, and 1024. `0` lets NVIDIA cuVS choose automatically. |
| `hashmap_mode` | `AUTO` | Hash map implementation used during search. Options include `HASH`, `SMALL`, and `AUTO`. |
| `hashmap_min_bitlen` | `0` | Lower limit for the hash map bit length. `0` lets NVIDIA cuVS choose automatically. |
| `hashmap_max_fill_rate` | `0.5` | Maximum hash map fill rate. Valid values are greater than 0.1 and less than 0.9. |
| `num_random_samplings` | `1` | Number of initial random seed-node selection iterations. |
| `rand_xor_mask` | `0x128394` | Bit mask used for initial random seed-node selection. |
| `persistent` | `False` | Uses the persistent search kernel where supported. Currently this applies only to `SINGLE_CTA`. |
| `persistent_lifetime` | `2.0` | Seconds before a persistent kernel stops when no requests are received. |
| `persistent_device_usage` | `1.0` | Fraction of the maximum grid size used by the persistent kernel. Lower values can leave GPU capacity for other work. |
| `filtering_rate` | `-1.0` | Expected fraction of nodes filtered out during filtered search. Negative values let NVIDIA cuVS estimate it automatically. |

## Tuning

The three parameters most often tuned are `itopk_size`, `graph_degree`, and `intermediate_graph_degree`.

Start with `itopk_size`. Increasing it usually improves recall, but lowers throughput because CAGRA keeps more candidates during search.

If search-time tuning is not enough, increase `graph_degree`. This gives each vertex more links to follow, but uses more memory and search work.

If the final graph quality is still too low, increase `intermediate_graph_degree`. This gives pruning more choices, but makes build more expensive.

### Persistent search

Persistent search can improve throughput in services that run many concurrent CAGRA searches. Instead of launching a new search kernel for each request, NVIDIA cuVS can keep a persistent search kernel resident on the GPU and feed it incoming work. This reduces launch overhead and can help high-volume search services keep the GPU busy.

Enable it with the `persistent` search parameter. Persistent search currently applies to the `SINGLE_CTA` search implementation, so set `algo` to `SINGLE_CTA` when you want to force this path instead of relying on `AUTO`.

Use `persistent_lifetime` to control how long the persistent kernel waits for new work before stopping. Use `persistent_device_usage` to reserve less than the full GPU for the persistent kernel when the same GPU also needs to run other kernels. Keeping other GPU work active alongside a persistent kernel can be fragile, so tune this setting carefully and validate it under the same concurrency pattern used in production.

## Memory footprint

CAGRA memory has two main parts: the dataset and the graph. During build, the dataset must be in GPU memory. After build, the dataset can be detached if it is not needed for search, for example when immediately converting the graph to HNSW.

To keep the formulas readable, this section uses short symbols. All estimates are in bytes. The examples convert bytes to MiB by dividing by `1024 * 1024`.

- `N`: Number of database vectors, or rows in the dataset being indexed.
- `D`: Vector dimension, or number of values in each vector.
- `B`: Bytes stored for each vector value. Use `4` for fp32, `2` for fp16, or the byte width of the attached dataset representation.
- `G`: Final graph degree. This is the `graph_degree` build parameter, and each vector keeps `G` neighbor IDs after pruning.
- `I`: Intermediate graph degree. This is the `intermediate_graph_degree` build parameter, and CAGRA uses this larger graph before pruning down to `G`.
- `C`: Number of IVF-PQ coarse clusters/lists. This is the IVF-PQ `n_lists` value used by the graph build parameters.
- `R`: IVF-PQ training-set ratio. This is `train_set_ratio`; `R = 10` means training uses roughly `N / 10` vectors.
- `Q`: Query batch size, or number of query vectors processed together.
- `K`: Search result count, or the requested `k`/`topk` nearest neighbors per query.
- `S_idx`: Bytes per graph neighbor ID. This is `sizeof(IdxT)`, usually `4` for `int32_t` or `uint32_t`.

The named terms in the formulas are also memory sizes:

- `dataset_size`: Device memory used by the attached dataset vectors.
- `graph_size`: Host memory used by the CAGRA graph neighbor IDs.
- `*_peak`: Temporary peak memory for one build phase. Sequential phases are not added together.
- `query_size`: Device memory for the current query batch.
- `result_size`: Device memory for neighbor IDs and distances returned for the current query batch.
- `workspace_size`: Query and result memory used during search.

### Scratch and maximum vectors

Most CAGRA formulas below are linear in `N` once build parameters are fixed. The named temporary peaks are the main scratch terms for build phases, but real runs can also include allocator padding, CUDA library workspaces, memory-resource pools, and small implementation buffers. Reserve a headroom factor `H = 0.20` for IVF-PQ graph builds and `H = 0.30` for NN-Descent or iterative-search graph builds. If you can measure a representative smaller run, use:

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


Choose the build or search formula that matches the operation, remove the explicit `scratch`/headroom from it, and rewrite it as:

$$
\text{peak\_without\_scratch}(N)
  = N \cdot B_{\text{per\_vector}} + B_{\text{fixed}}
$$

Then estimate:

$$
N_{\max}
  =
  \left\lfloor
    \frac{M_{\text{usable}} - B_{\text{fixed}}}
         {B_{\text{per\_vector}}}
  \right\rfloor
$$

For out-of-core IVF-PQ graph build, `Q`, `C`, and `R` can make several terms fixed or sublinear for a fixed configuration. Solve the full `max(...)` expression if the largest phase changes as `N` changes.

### Baseline memory after build

The baseline memory footprint after index construction is:

$$
\begin{aligned}
\text{dataset\_size (device)}
&= N \times D \times B
\end{aligned}
$$

$$
\begin{aligned}
\text{graph\_size (host)}
&= N \times G \times S_{\text{idx}}
\end{aligned}
$$

The dataset must be in GPU memory during index build, but can be detached afterward if it is not needed for search.

**Example** (1,000,000 vectors, dim = 1024, fp32, `graph_degree = 64`, `IdxT = int32`):

- `dataset_size = 4,096,000,000 B = 3906.25 MB`
- `graph_size = 256,000,000 B = 244.14 MB`

### Build peak memory usage

Index build has two phases: construct an initial kNN graph, then optimize it by pruning redundant paths. These steps run sequentially, so their peak memory use is not additive. The overall peak depends on the configured RMM memory resource.

The initial graph can be built with IVF-PQ, NN-Descent, or the experimental iterative CAGRA-search builder. IVF-PQ can build in batches, which allows CAGRA to train on datasets larger than available GPU memory. The iterative builder requires the aligned dataset to fit in GPU memory because it repeatedly searches the partially built CAGRA graph.

### Initial graph build using IVF-PQ

IVF-PQ builds the initial graph in two stages. First, it trains cluster centroids and PQ codebooks. Then it queries the IVF-PQ index in batches to form approximate nearest-neighbor lists.

**IVF-PQ build peak:**

Here, `N / R` is the IVF-PQ training sample size. The `4` byte factors are fp32 values for training vectors and cluster centroids. The `uint32_t` term stores one 32-bit ID per training vector.

$$
\begin{aligned}
\text{IVFPQ\_build\_peak}
&= \frac{N}{R} \times D \times 4 \\
&\quad + C \times D \times 4 \\
&\quad + \frac{N}{R}
  \times \operatorname{sizeof}(\mathrm{uint32\_t})
\end{aligned}
$$

**Example** (`N = 1e6`, `D = 1024`, `C = 1024`, `R = 10`): 395.01 MB

**IVF-PQ search peak:**

Here, `Q` is the number of vectors in one search batch and `I` is the number of candidates kept per query while building the intermediate graph. The three terms estimate query vectors, candidate IDs, and candidate distances.

$$
\begin{aligned}
\text{IVFPQ\_search\_peak}
&= Q \times D \times 4 \\
&\quad + Q \times I
  \times \operatorname{sizeof}(\mathrm{uint32\_t}) \\
&\quad + Q \times I \times 4
\end{aligned}
$$

**Example** (`Q = 1024`, `D = 1024`, `I = 128`): 5.00 MB

### Initial graph build using NN-Descent

**Peak device memory:**

The constants in the NN-Descent formulas are per-vector workspace estimates from the implementation. They are added to the vector storage terms before multiplying by `N`.

$$
\begin{aligned}
\text{NND\_device\_peak}
&= N \times (D \times 2 + 276)
\end{aligned}
$$

- Data vectors are transferred to device and stored as fp16: `D * 2` bytes per vector.
- The small working graph, locks, and edge counters use 276 bytes per vector.
- L2 metric adds 4 bytes per vector for precomputed norms.

**Peak host memory:**

$$
\begin{aligned}
\text{NND\_host\_peak}
&= N \times (13 \times I + 912)
\end{aligned}
$$

- Full graph with distances: `1.3 * 8 * I` bytes per vector.
- Bloom filter for sampling: `1.3 * 2 * I` bytes per vector.
- 5 sample buffers with degree 32: 640 bytes per vector.
- Graph update buffer with degree 32: 256 bytes per vector.
- Edge counters: 16 bytes per vector.

### Initial graph build using iterative CAGRA search

The iterative builder starts with a small connected graph, then repeatedly uses CAGRA search to find neighbors for a larger prefix of the dataset. After each search pass, it optimizes the graph and doubles the active graph size until all rows are included.

This path is useful when the metric or data type is better served by CAGRA search itself, but it is not an out-of-core builder. The dataset is copied or aligned into GPU memory before the first iteration.

Variables used only in this subsection:

- `D_align`: Aligned device stride used by CAGRA search. Use `D` when no padding is required.
- `Q_iter`: Maximum query chunk size used by the iterative builder. The implementation currently uses `min(N, 8192)`.
- `K_iter`: Number of temporary neighbors kept per query during the last pass. Use `I + 1`.
- `G_iter`: Largest graph degree used by the temporary searchable graph. Use `G`; early iterations use a smaller degree and the final iterations use `G`.
- `D_iter`: Aligned device dataset memory.
- `G_tmp`: Largest temporary device graph memory.
- `Q_tile`: Query tile memory for one search chunk.
- `R_tile`: Result tile memory for one search chunk.
- `W_iter`: Temporary device workspace used by one iterative search pass.
- `H_iter`: Host neighbors-list capacity in bytes after rounding up to a 2 MiB boundary. One MiB is `1024 * 1024` bytes.

The aligned device dataset is:

$$
\begin{aligned}
D_{\text{iter}}
&= N \times D_{\text{align}} \times B
\end{aligned}
$$

The largest temporary device graph used during the search pass is:

$$
\begin{aligned}
G_{\text{tmp}}
&= N \times G_{\text{iter}} \times S_{\text{idx}}
\end{aligned}
$$

Each search chunk needs query storage plus temporary neighbor IDs and distances:

$$
\begin{aligned}
Q_{\text{tile}}
&= Q_{\text{iter}} \times D_{\text{align}} \times B \\
R_{\text{tile}}
&= Q_{\text{iter}} \times K_{\text{iter}}
   \times (S_{\text{idx}} + 4)
\end{aligned}
$$

The host neighbors list stores the temporary neighbor candidates for all rows:

$$
\begin{aligned}
H_{\text{iter}}
&=
\operatorname{round\_up}
\big(
  N \times K_{\text{iter}} \times S_{\text{idx}},
  2\ \text{MiB}
\big)
\end{aligned}
$$

The temporary device workspace for one search pass is:

$$
\begin{aligned}
W_{\text{iter}}
&= G_{\text{tmp}} \\
&\quad + Q_{\text{tile}} \\
&\quad + R_{\text{tile}}
\end{aligned}
$$

The practical device peak for the iterative graph build is:

$$
\begin{aligned}
\text{iterative\_device\_peak}
&\approx
  D_{\text{iter}} \\
&\quad + \max\!\big(
  W_{\text{iter}},
  \text{optimize\_peak}
  \big)
\end{aligned}
$$

The practical host peak is:

$$
\begin{aligned}
\text{iterative\_host\_peak}
&\approx
  H_{\text{iter}}
  + N \times G \times S_{\text{idx}}
\end{aligned}
$$

The final `N * G * S_idx` term is the host graph that remains after build. Check device and host memory separately. The usable `N` is the smaller value allowed by `iterative_device_peak` and `iterative_host_peak`.

### Optimize phase

The optimize phase prunes and reorders the intermediate graph. Its peak memory scales linearly with the intermediate degree:

In this formula, the `4` byte term is per-vector bookkeeping. The `(S_idx + 1) * I` term stores `I` candidate neighbor IDs plus one byte of pruning state per candidate.

$$
\begin{aligned}
\text{optimize\_peak}
&= N \times
  \Big( 4 + (S_{\text{idx}} + 1) \times I \Big)
\end{aligned}
$$

**Example** (`N = 1e6`, `I = 128`, `IdxT = int32`): 614.17 MB

Out-of-core CAGRA build consists of IVF-PQ build, IVF-PQ search, and CAGRA optimization. These steps are sequential, so their temporary memory peaks are not added together.

### Overall build peak memory usage

The overall device peak is the dataset size plus the largest temporary allocation from the sequential build steps.

**Using IVF-PQ:**

$$
\begin{aligned}
\text{build\_peak}
&= \text{dataset\_size} \\
&\quad + \max\!\big(
  \text{IVFPQ\_build\_peak}, \\
&\qquad\qquad
  \text{IVFPQ\_search\_peak}, \\
&\qquad\qquad
  \text{optimize\_peak}
\big)
\end{aligned}
$$

**Example:** `3906.25 + max(395.01, 5.00, 614.17) = 4520.42 MB`

**Using NN-Descent:**

$$
\begin{aligned}
\text{build\_peak}
&= \text{dataset\_size}^{*} \\
&\quad + \max\!\big(
  \text{NND\_device\_peak}, \\
&\qquad\qquad
  \text{optimize\_peak}
\big)
\end{aligned}
$$

`dataset_size*` applies only when the user passes data that is already in device memory. NN-Descent internally copies the dataset to the device as fp16, so host-memory inputs do not add this term.

**Using iterative CAGRA search:**

Use `iterative_device_peak` for device memory and `iterative_host_peak` for host memory. These estimates already include the aligned dataset, temporary search chunks, temporary graph storage, optimization workspace, and final host graph.

## Search peak memory usage

CAGRA search requires the dataset and graph to already be resident in GPU memory. When using CAGRA-Q, the original dataset can reside in host memory instead. Search also needs temporary workspace for query vectors and results.

If multiple batches run concurrently or overlap, each batch needs separate result buffers. The estimate below assumes one query batch at a time and reused buffers.

$$
\begin{aligned}
\text{search\_memory}
&= \text{dataset\_size} \\
&\quad + \text{graph\_size} \\
&\quad + \text{workspace\_size}
\end{aligned}
$$

The workspace contains query vectors and result storage:

In the query formula, `sizeof(float)` is `4` bytes because CAGRA search uses fp32 query storage here. In the result formula, each returned neighbor stores one graph ID of size `S_idx` and one fp32 distance.

$$
\begin{aligned}
\text{query\_size}
&= Q \times D
  \times \operatorname{sizeof}(\mathrm{float})
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

$$
\begin{aligned}
\text{workspace\_size}
&= \text{query\_size}
  + \text{result\_size}
\end{aligned}
$$

**Example** (`D = 1024`, `Q = 100`, `K = 10`, `IdxT = int32`):

- `query_size = 409,600 B = 0.39 MB`
- `result_size = 8,000 B = 0.0076 MB`
- `workspace_size = query_size + result_size = 0.40 MB`
- `total search memory ~= 3906.25 + 244.14 + 0.40 = 4150.79 MB`
