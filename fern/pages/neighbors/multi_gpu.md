# Multi-GPU

Multi-GPU indexing is a coordination layer for supported nearest-neighbor indexes. It does not define a new index family. Instead, it lets APIs such as [CAGRA](/user-guide/api-guides/indexing-guide/cagra), [IVF-Flat](/user-guide/api-guides/indexing-guide/ivf-flat), and [IVF-PQ](/user-guide/api-guides/indexing-guide/ivf-pq) build, store, search, extend, serialize, and deserialize indexes across multiple GPUs.

Use multi-GPU indexing when a single GPU is not enough for the desired dataset size, build speed, or query throughput. The same index-specific parameters still matter; multi-GPU adds resource management, distribution mode, and result-merging behavior on top.

## Example API Usage

[C API common types](/api-reference/c-api-neighbors-mg-common) | [C API resources](/api-reference/c-api-core-c-api#cuvsmultigpuresourcescreate) | [C API CAGRA](/api-reference/c-api-neighbors-mg-cagra) | [C API IVF-Flat](/api-reference/c-api-neighbors-mg-ivf-flat) | [C API IVF-PQ](/api-reference/c-api-neighbors-mg-ivf-pq) | [Python CAGRA](/api-reference/python-api-neighbors-mg-cagra) | [Python IVF-Flat](/api-reference/python-api-neighbors-mg-ivf-flat) | [Python IVF-PQ](/api-reference/python-api-neighbors-mg-ivf-pq)

The examples below use CAGRA, but the same pattern applies to IVF-Flat and IVF-PQ: create multi-GPU resources, wrap the index-specific build and search parameters with multi-GPU settings, then pass host matrices to the multi-GPU build and search functions.

### Building and searching an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/mg_cagra.h>

cuvsResources_t res;
cuvsMultiGpuCagraIndexParams_t index_params;
cuvsMultiGpuCagraSearchParams_t search_params;
cuvsMultiGpuCagraIndex_t index;
DLManagedTensor *dataset;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

// Populate host tensors for dataset, queries, neighbors, and distances.
load_host_dataset(dataset);
load_host_queries(queries);
allocate_host_outputs(neighbors, distances);

cuvsMultiGpuResourcesCreate(&res);
cuvsMultiGpuCagraIndexParamsCreate(&index_params);
cuvsMultiGpuCagraSearchParamsCreate(&search_params);
cuvsMultiGpuCagraIndexCreate(&index);

index_params->mode = CUVS_NEIGHBORS_MG_SHARDED;
search_params->merge_mode = CUVS_NEIGHBORS_MG_TREE_MERGE;

cuvsMultiGpuCagraBuild(res, index_params, dataset, index);
cuvsMultiGpuCagraSearch(res, search_params, index, queries, neighbors, distances);

cuvsMultiGpuCagraIndexDestroy(index);
cuvsMultiGpuCagraSearchParamsDestroy(search_params);
cuvsMultiGpuCagraIndexParamsDestroy(index_params);
cuvsMultiGpuResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_resources_snmg.hpp>

using namespace cuvs::neighbors;

raft::device_resources_snmg clique;
auto dataset = load_host_dataset<float, int64_t>();
auto queries = load_host_queries<float, int64_t>();
auto neighbors = allocate_host_neighbors<int64_t, int64_t>();
auto distances = allocate_host_distances<float, int64_t>();

mg_index_params<cagra::index_params> index_params;
index_params.mode = cuvs::neighbors::SHARDED;
index_params.metric = cuvs::distance::DistanceType::L2Expanded;

auto index = cagra::build(clique, index_params, dataset.view());

mg_search_params<cagra::search_params> search_params;
search_params.merge_mode = cuvs::neighbors::TREE_MERGE;
search_params.n_rows_per_batch = 1 << 20;

cagra::search(clique,
              index,
              search_params,
              queries.view(),
              neighbors.view(),
              distances.view());
```

</Tab>
<Tab title="Python">

```python
import numpy as np

from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import cagra

dataset = np.asarray(load_dataset(), dtype=np.float32)
queries = np.asarray(load_queries(), dtype=np.float32)

resources = MultiGpuResources()
resources.set_memory_pool(80)

index_params = cagra.IndexParams(
    distribution_mode="sharded",
    metric="sqeuclidean",
)
index = cagra.build(index_params, dataset, resources=resources)

search_params = cagra.SearchParams(
    merge_mode="tree_merge",
    n_rows_per_batch=1000,
)
distances, neighbors = cagra.search(
    search_params,
    index,
    queries,
    k=10,
    resources=resources,
)

resources.sync()
```

</Tab>
</Tabs>

### Extending an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/mg_cagra.h>

cuvsResources_t res;
cuvsMultiGpuCagraIndex_t index;
DLManagedTensor *new_vectors;
DLManagedTensor *new_indices;

load_host_additional_dataset(new_vectors);
load_host_additional_indices(new_indices);

cuvsMultiGpuResourcesCreate(&res);
cuvsMultiGpuCagraIndexCreate(&index);

// ... build or load index ...
cuvsMultiGpuCagraExtend(res, index, new_vectors, new_indices);

cuvsMultiGpuCagraIndexDestroy(index);
cuvsMultiGpuResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_resources_snmg.hpp>

using namespace cuvs::neighbors;

raft::device_resources_snmg clique;
mg_index_params<cagra::index_params> index_params;
auto dataset = load_host_dataset<float, int64_t>();
auto new_vectors = load_host_additional_dataset<float, int64_t>();

auto index = cagra::build(clique, index_params, dataset.view());
cagra::extend(clique, index, new_vectors.view(), std::nullopt);
```

</Tab>
<Tab title="Python">

```python
import numpy as np

from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import cagra

dataset = np.asarray(load_dataset(), dtype=np.float32)
new_vectors = np.asarray(load_additional_data(), dtype=np.float32)
new_indices = np.asarray(load_additional_indices(), dtype=np.uint32)

resources = MultiGpuResources()
index = cagra.build(cagra.IndexParams(), dataset, resources=resources)
cagra.extend(index, new_vectors, new_indices, resources=resources)
```

</Tab>
</Tabs>

The same pattern applies to multi-GPU IVF-Flat and IVF-PQ. See the [CAGRA C](/api-reference/c-api-neighbors-mg-cagra#cuvsmultigpucagraextend), [IVF-Flat C](/api-reference/c-api-neighbors-mg-ivf-flat#cuvsmultigpuivfflatextend), [IVF-PQ C](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqextend), [CAGRA Python](/api-reference/python-api-neighbors-mg-cagra#extend), [IVF-Flat Python](/api-reference/python-api-neighbors-mg-ivf-flat#extend), and [IVF-PQ Python](/api-reference/python-api-neighbors-mg-ivf-pq#extend) API references for the full signatures.

## How Multi-GPU indexing works

The multi-GPU APIs use a resource object that owns the participating GPU set and the communication state needed by the operation. In C++, this is typically a `raft::device_resources_snmg`. In C and Python, create the corresponding NVIDIA cuVS multi-GPU resources object.

The build parameters extend the single-GPU index parameters. For example, a multi-GPU CAGRA build parameter object contains normal CAGRA settings plus a distribution mode. This means you tune graph degree, IVF list count, PQ settings, metrics, and other index-specific choices in the same way as the single-GPU guide, then decide how the index should be placed across GPUs.

The search parameters also extend the single-GPU search parameters. Multi-GPU search adds controls for how queries are assigned to replicas, how shard results are merged, and how many query rows are processed per batch.

## Supported indexes

| Index | Multi-GPU support | Notes |
| --- | --- | --- |
| [CAGRA](/user-guide/api-guides/indexing-guide/cagra) | C, C++, Python | Graph index build, search, extend, serialization, deserialization, and single-GPU index distribution are exposed through the multi-GPU APIs. |
| [IVF-Flat](/user-guide/api-guides/indexing-guide/ivf-flat) | C, C++, Python | Full-precision inverted-file indexes can be sharded or replicated across GPUs. |
| [IVF-PQ](/user-guide/api-guides/indexing-guide/ivf-pq) | C, C++, Python | Compressed inverted-file indexes can be sharded or replicated across GPUs. |
| [All-neighbors](/user-guide/api-guides/indexing-guide/all-neighbors) | Python resource support | The Python all-neighbors API can use `MultiGpuResources`, but it is not exposed as a `mg_index` wrapper. |

## Distribution modes

| Mode | What it does | When to use it |
| --- | --- | --- |
| `sharded` | Splits the index across GPUs. Each query searches the participating shards, and partial results are merged. | Use this when the index is too large for one GPU or when you want capacity to scale with the number of GPUs. |
| `replicated` | Copies the full index to each GPU. Queries are assigned across replicas. | Use this when the full index fits on each GPU and throughput is the main goal. |

Sharding favors scale because each GPU owns only part of the index. Replication favors serving throughput because each GPU can answer different query work independently.

## Configuration parameters

### Build parameters

The multi-GPU build parameter wrapper contains the selected index's normal build parameters plus one multi-GPU field.

| Parameter | Values | Description |
| --- | --- | --- |
| `mode` / `distribution_mode` | `sharded`, `replicated` | Controls whether the index is split across GPUs or copied to each GPU. The default is sharded. |

Use the base parameter object for the underlying index-specific settings. For CAGRA, tune the fields described in [CAGRA build parameters](/user-guide/api-guides/indexing-guide/cagra#build-parameters). For IVF-Flat and IVF-PQ, tune the fields described in their corresponding guide pages.

### Search parameters

The multi-GPU search parameter wrapper contains the selected index's normal search parameters plus fields that control multi-GPU scheduling and merging.

| Parameter | Values | Description |
| --- | --- | --- |
| `search_mode` | `load_balancer`, `round_robin` | Used for replicated indexes. Load balancing assigns work to keep GPUs busy; round robin assigns query batches in order. |
| `merge_mode` | `merge_on_root_rank`, `tree_merge` | Used for sharded indexes. Root merge gathers all shard results on one rank; tree merge reduces results in stages. |
| `n_rows_per_batch` | Positive integer | Number of query rows processed in each multi-GPU batch. Larger batches can improve throughput but require more temporary memory. |

## Tuning

Start by tuning the single-GPU index parameters on one representative GPU. Once the index quality and single-GPU latency are reasonable, choose a multi-GPU distribution mode.

Use `sharded` mode when capacity is the limiting factor. Increase `n_rows_per_batch` until throughput stops improving or temporary memory becomes too high. Prefer `tree_merge` for larger GPU counts because it avoids concentrating all merge work on one rank.

Use `replicated` mode when the full index fits on each GPU and the workload has many independent queries. Compare `load_balancer` and `round_robin`; load balancing is usually the safer default when query batch sizes or query costs vary.

## Memory footprint

Multi-GPU memory depends on the selected index and distribution mode. Use the memory formulas on the single-GPU guide page for the underlying index, then apply the placement model below.

Let:

- `G` be the number of GPUs.
- `N` be the total number of indexed vectors.
- `D` be the vector dimensionality.
- `M_index(N, D)` be the single-GPU index footprint for the selected index.
- `Q_b` be `n_rows_per_batch`.
- `k` be the number of neighbors returned per query.
- `S_result` be `sizeof(index_type) + sizeof(distance_type)`.

For a replicated index, each GPU stores a full copy:

```text
per_gpu_index_memory ~= M_index(N, D)
total_index_memory ~= G * M_index(N, D)
```

For a sharded index with balanced shards, each GPU stores about one `G`th of the index:

```text
per_gpu_index_memory ~= M_index(ceil(N / G), D)
total_index_memory ~= sum(per-shard index memory)
```

Search also needs output and merge space. A useful planning estimate for merge state is:

```text
merge_memory ~= G * Q_b * k * S_result
```

This estimate is intentionally conservative. Actual peak memory depends on the selected index, datatype, batch size, distribution balance, and whether merge work is done on one root rank or in a tree.

## Saving, loading, and distributing indexes

The multi-GPU C, C++, and Python APIs expose serialization and deserialization for supported indexes. CAGRA, IVF-Flat, and IVF-PQ also expose a distribution path that loads a local single-GPU index file and creates a multi-GPU index from it.

Use serialization when you want to persist the multi-GPU index layout. Use distribution when you already have a single-GPU index artifact and want NVIDIA cuVS to place it across the multi-GPU resources for serving.
