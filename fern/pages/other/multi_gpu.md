# Multi-GPU

NVIDIA cuVS multi-GPU APIs use RAFT resources to coordinate work across GPUs. The resource object owns CUDA streams, memory resources, and communication state, so NVIDIA cuVS algorithms can be written against one interface and then run in different distributed environments.

The RAFT communicator is the part of that interface that handles rank metadata and collective communication. This lets an algorithm use the same communication pattern whether the surrounding application is launched with MPI, Dask, Ray, or another distributed runtime. The runtime is still responsible for starting workers, assigning ranks, and placing data shards; RAFT gives NVIDIA cuVS a common way to communicate once those pieces exist.

NCCL is the primary communicator backend used by NVIDIA cuVS multi-GPU algorithms. Most users interact with NCCL through one of two paths:

- Single-node multi-GPU resources, where one process controls multiple GPUs on the same node.
- Multi-node multi-GPU resources, where each process owns a rank and attaches an externally created NCCL communicator to a RAFT handle.

For multi-GPU vector indexes, see the [Multi-GPU indexing guide](/user-guide/api-guides/indexing-guide/multi-gpu).

## Example API Usage

[C resources API](/api-reference/c-api-core-c-api#cuvsmultigpuresourcescreate) | [Python resources API](/api-reference/python-api-common#multigpuresources)

The examples below cover the high-level NVIDIA cuVS language surfaces that currently expose multi-GPU resource initialization: C, C++, and Python. Rust, Go, and Java do not currently expose matching high-level multi-GPU resource wrappers.

### Single-node multi-GPU

Use the single-node path when one process can see and control all GPUs used by the operation. This is the simplest setup for one machine with multiple GPUs. In C++ this is `raft::device_resources_snmg`; in C and Python it is exposed through NVIDIA cuVS multi-GPU resources wrappers.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>

cuvsResources_t resources;
cuvsMultiGpuResourcesCreate(&resources);

// Use resources with cuVS multi-GPU C APIs.
// For example, pass it to cuvsMultiGpuCagraBuild().

cuvsMultiGpuResourcesDestroy(resources);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_resources_snmg.hpp>

using namespace cuvs::neighbors;

raft::device_resources_snmg resources;
auto dataset = load_host_dataset<float, int64_t>();

mg_index_params<cagra::index_params> index_params;
index_params.mode = cuvs::neighbors::SHARDED;

auto index = cagra::build(resources, index_params, dataset.view());
```

</Tab>
<Tab title="Python">

```python
import numpy as np

from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import cagra

dataset = np.asarray(load_dataset(), dtype=np.float32)

resources = MultiGpuResources(device_ids=[0, 1])
resources.set_memory_pool(80)

index_params = cagra.IndexParams(distribution_mode="sharded")
index = cagra.build(index_params, dataset, resources=resources)

resources.sync()
```

</Tab>
</Tabs>

When an application should restrict NVIDIA cuVS to a subset of visible GPUs, use the device-id-specific resource constructor for that language:

- C: `cuvsMultiGpuResourcesCreateWithDeviceIds()`
- C++: `raft::device_resources_snmg(std::vector<int>{...})`
- Python: `MultiGpuResources(device_ids=[...])`

### Multi-node NCCL communicator

Use the multi-node path when each process controls one rank, often one GPU, and the application runtime provides launch, rank assignment, and data placement. This API is currently exposed only in C++. The application creates an `ncclComm_t`, attaches it to a RAFT handle, and then passes that handle to NVIDIA cuVS APIs that accept `raft::resources`.

<Tabs>
<Tab title="C++">

```cpp
#include <cuvs/cluster/kmeans.hpp>

#include <raft/comms/std_comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>

#include <cuda_runtime_api.h>
#include <mpi.h>
#include <nccl.h>

#include <optional>

using namespace cuvs::cluster;

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int device_count;
  cudaGetDeviceCount(&device_count);
  cudaSetDevice(rank % device_count);

  ncclUniqueId nccl_id;
  if (rank == 0) { ncclGetUniqueId(&nccl_id); }
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t nccl_comm;
  ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank);

  raft::handle_t handle;
  raft::comms::build_comms_nccl_only(&handle, nccl_comm, world_size, rank);

  // Each rank owns one local shard on its GPU.
  auto local_dataset = load_local_dataset<float, int>(rank);

  kmeans::params params;
  params.n_clusters = 1024;

  auto centroids =
      raft::make_device_matrix<float, int>(
          handle, params.n_clusters, local_dataset.extent(1));

  float inertia;
  int n_iter;

  kmeans::fit(handle,
              params,
              local_dataset.view(),
              std::nullopt,
              centroids.view(),
              raft::make_host_scalar_view(&inertia),
              raft::make_host_scalar_view(&n_iter));

  handle.sync_stream();

  ncclCommDestroy(nccl_comm);
  MPI_Finalize();
  return 0;
}
```

</Tab>
</Tabs>

The example uses MPI only to launch ranks and broadcast the NCCL unique ID. A Ray, Dask, or service-based runtime can provide the same rank metadata and NCCL communicator setup through its own worker lifecycle.

## RAFT communicator role

The communicator makes distributed NVIDIA cuVS code less tied to one scheduler. NVIDIA cuVS algorithms call collectives through RAFT resources instead of embedding MPI, Dask, or Ray-specific logic in the algorithm itself. This is what allows the same algorithm implementation to be reused in different deployment systems.

In practice, the communicator provides:

- The rank id and world size for the current worker.
- Collective operations used by distributed algorithms.
- A common place for NVIDIA cuVS to find communication state alongside CUDA streams and memory resources.

NCCL is the communicator used for GPU collectives in NVIDIA cuVS. MPI, Ray, Dask, or another framework may still be used to launch workers, distribute data, and exchange the NCCL unique ID before the RAFT handle is initialized.

## Choosing a setup

| Setup | Languages | Typical use | Who creates communication state? |
| --- | --- | --- | --- |
| Single&#8209;node&nbsp;multi&#8209;GPU | C, C++, Python | One process uses several GPUs on one machine. | NVIDIA cuVS/RAFT creates resources for the visible devices or requested device ids. |
| Multi&#8209;node&nbsp;multi&#8209;GPU | C++ | Many ranks run across one or more nodes. | The application runtime creates ranks and initializes NCCL. |

Use single-node resources when all GPUs are local to one process. Use an explicit NCCL communicator when the work is already distributed across ranks, nodes, or worker processes.
