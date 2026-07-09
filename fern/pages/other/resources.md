# Resources

NVIDIA cuVS APIs use a resources object to keep track of CUDA execution state that should be reused across related calls. In C++ this is usually `raft::device_resources`; in C and most language bindings it appears as a wrapper around `cuvsResources_t`.

For simple examples, the default resource behavior is usually enough. Create and pass an explicit resources object when you want to chain several operations together, control the CUDA stream used by NVIDIA cuVS, reuse expensive CUDA library handles, configure temporary memory, or keep setup costs out of repeated calls.

## Common Concepts

A [CUDA stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) is an ordered queue of GPU work. Kernel launches, copies, and library calls enqueued into the same stream run in order. Work in different streams can overlap when the GPU and workload allow it.

Most NVIDIA cuVS algorithms enqueue work on the stream stored in the resources object and return control to the host before that GPU work has necessarily finished. This is intentional: it lets users chain operations without forcing a synchronization point between every call.

Synchronization means waiting until queued GPU work has completed. CUDA provides [explicit synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#explicit-synchronization), and the CUDA Runtime API includes `cudaStreamSynchronize()` for waiting on one stream. NVIDIA cuVS resource wrappers expose stream synchronization helpers so users do not need to fetch the raw CUDA stream in common cases.

A stream pool is a small group of CUDA streams attached to a resources object. Some algorithms can use a stream pool to run independent pieces of work concurrently. Most users do not need to configure one unless an algorithm guide or benchmark suggests it.

Workspace memory is temporary memory used inside an algorithm. Configuring workspace resources can make allocation behavior more predictable for allocation-heavy workloads. See [Memory Management](/user-guide/api-guides/core-types/memory-management) for allocator choices such as device pools, pinned host memory, managed memory, and host memory.

## Example API Usage

[C resources API](/api-reference/c-api-core-c-api#cuvsresourcescreate) | [Python resources API](/api-reference/python-api-common#resources) | [Java resources API](/api-reference/java-api-com-nvidia-cuvs-cuvsresources) | [Rust resources API](/api-reference/rust-api-cuvs-resources) | [Go resources API](/api-reference/go-api-cuvs#resource)

### Creating and reusing resources

Create one resources object near the beginning of a workflow and pass it to related NVIDIA cuVS calls. Reusing the same object keeps those calls ordered on the same CUDA stream and lets NVIDIA cuVS reuse expensive state.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>

int main(void)
{
  cuvsResources_t resources;

  if (cuvsResourcesCreate(&resources) != CUVS_SUCCESS) { return 1; }

  // Pass resources to cuVS C APIs, such as index build and search calls.

  if (cuvsStreamSync(resources) != CUVS_SUCCESS) { return 1; }
  if (cuvsResourcesDestroy(resources) != CUVS_SUCCESS) { return 1; }

  return 0;
}
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace brute_force = cuvs::neighbors::brute_force;

raft::device_resources resources;

auto dataset = load_device_dataset_view<float, int64_t>();
auto index = brute_force::build(resources, brute_force::index_params{}, dataset);

auto queries = load_device_query_view<float, int64_t>();
auto neighbors = make_device_neighbors_view<int64_t, int64_t>();
auto distances = make_device_distances_view<float, int64_t>();

brute_force::search(resources,
                    brute_force::search_params{},
                    index,
                    queries,
                    neighbors,
                    distances);

raft::resource::sync_stream(resources);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp

from cuvs.common import Resources
from cuvs.neighbors import brute_force

resources = Resources()

dataset = cp.asarray(load_dataset(), dtype=cp.float32)
queries = cp.asarray(load_queries(), dtype=cp.float32)

index = brute_force.build(dataset, resources=resources)
distances, neighbors = brute_force.search(index, queries, k=10, resources=resources)

resources.sync()
```

</Tab>
<Tab title="Java">

```java
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;

try (CuVSResources resources = CuVSResources.create();
    CuVSMatrix dataset = loadDatasetMatrix()) {
  BruteForceIndexParams params = new BruteForceIndexParams.Builder().build();

  try (BruteForceIndex index =
      BruteForceIndex.newBuilder(resources)
          .withDataset(dataset)
          .withIndexParams(params)
          .build()) {
    // Reuse resources with search, serialization, or other cuVS calls.
  }
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::resources::Resources;
use cuvs::Result;

fn run() -> Result<()> {
    let resources = Resources::new()?;

    // Pass &resources to cuVS Rust APIs.

    resources.sync_stream()?;
    Ok(())
}
```

</Tab>
<Tab title="Go">

```go
package main

import cuvs "github.com/rapidsai/cuvs/go"

func main() error {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		return err
	}
	defer resource.Close()

	// Pass resource to cuVS Go APIs.

	return resource.Sync()
}
```

</Tab>
</Tabs>

### Synchronizing GPU work

Most NVIDIA cuVS algorithms do not synchronize before returning. This lets you enqueue several operations back-to-back on the same resources object, such as build, search, refine, and copy, without stopping the GPU between steps.

Synchronize explicitly before reading results on the host, measuring elapsed time, handing data to work on a different CUDA stream without another dependency, or destroying objects that may still be used by queued GPU work.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>

cuvsResources_t resources;
cuvsResourcesCreate(&resources);

// Enqueue one or more cuVS C API calls on resources.

cuvsStreamSync(resources);
cuvsResourcesDestroy(resources);
```

</Tab>
<Tab title="C++">

```cpp
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

raft::device_resources resources;

// Enqueue one or more cuVS C++ API calls on resources.

raft::resource::sync_stream(resources);
```

</Tab>
<Tab title="Python">

```python
from cuvs.common import Resources

resources = Resources()

# Pass resources=resources to one or more cuVS Python calls.

resources.sync()
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::resources::Resources;
use cuvs::Result;

fn run() -> Result<()> {
    let resources = Resources::new()?;

    // Pass &resources to one or more cuVS Rust calls.

    resources.sync_stream()?;
    Ok(())
}
```

</Tab>
<Tab title="Go">

```go
package main

import cuvs "github.com/rapidsai/cuvs/go"

func syncWork(resource cuvs.Resource) error {
	// Call cuVS Go APIs with resource first.
	return resource.Sync()
}
```

</Tab>
</Tabs>

In Python, many APIs create and synchronize an internal resources object when you do not pass one. When you pass your own `Resources`, you are also taking responsibility for calling `resources.sync()` at the point where your application needs the results to be complete.

## Multi-threaded Applications

`raft::resources` and the resource wrappers built on top of it are not thread-safe for concurrent use. If an application uses multiple host threads, give each worker thread its own resources object. This keeps each thread's CUDA stream, library handles, temporary memory, and queued work separate.

Do not share one resources object across threads unless the application uses its own locking and understands that the lock serializes access to that resource. For most workloads, one resources object per worker thread is simpler and faster.

<Tabs>
<Tab title="C++">

```cpp
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <thread>
#include <vector>

void worker(int worker_id)
{
  raft::device_resources resources;

  // Run this thread's cuVS work with its own resources object.
  run_cuvs_workload(worker_id, resources);

  raft::resource::sync_stream(resources);
}

int main()
{
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back(worker, i);
  }

  for (auto& thread : threads) {
    thread.join();
  }
}
```

</Tab>
<Tab title="Python">

```python
from concurrent.futures import ThreadPoolExecutor

from cuvs.common import Resources


def worker(worker_id):
    resources = Resources()

    # Run this thread's cuVS work with its own Resources object.
    run_cuvs_workload(worker_id, resources=resources)

    resources.sync()


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(worker, range(4)))
```

</Tab>
<Tab title="Java">

```java
import com.nvidia.cuvs.CuVSResources;

Runnable worker =
    () -> {
      try (CuVSResources resources = CuVSResources.create()) {
        // Run this thread's cuVS work with its own CuVSResources object.
        runCuVSWorkload(resources);
      } catch (Throwable error) {
        throw new RuntimeException(error);
      }
    };

Thread first = new Thread(worker);
Thread second = new Thread(worker);

first.start();
second.start();
first.join();
second.join();
```

</Tab>
</Tabs>

## Using an Application CUDA Stream

Most users can let NVIDIA cuVS create and own the CUDA stream inside the resources object. Supplying an application stream is useful when NVIDIA cuVS work must be ordered with other CUDA kernels, copies, or library calls from the same application.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>
#include <cuda_runtime_api.h>

cudaStream_t stream;
cuvsResources_t resources;

cudaStreamCreate(&stream);
cuvsResourcesCreate(&resources);
cuvsStreamSet(resources, stream);

// cuVS calls using resources are enqueued on stream.

cuvsStreamSync(resources);
cuvsResourcesDestroy(resources);
cudaStreamDestroy(stream);
```

</Tab>
<Tab title="C++">

```cpp
#include <raft/core/device_resources.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

cudaStream_t stream;
cudaStreamCreate(&stream);

raft::device_resources resources{rmm::cuda_stream_view{stream}};

// cuVS C++ calls using resources are enqueued on stream.

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp

from cuvs.common import Resources

stream = cp.cuda.Stream(non_blocking=True)
resources = Resources(stream=stream.ptr)

with stream:
    # cuVS calls using resources are ordered with work in this CuPy stream.
    pass

resources.sync()
```

</Tab>
<Tab title="Rust">

```rust
use cuvs::resources::Resources;
use cuvs_sys::cudaStream_t;

fn configure_stream(stream: cudaStream_t) -> cuvs::Result<Resources> {
    let resources = Resources::new()?;
    resources.set_cuda_stream(stream)?;
    Ok(resources)
}
```

</Tab>
<Tab title="Go">

```go
package main

import "C"
import cuvs "github.com/rapidsai/cuvs/go"

func newResourceOnStream(stream C.cudaStream_t) (cuvs.Resource, error) {
	return cuvs.NewResource(stream)
}
```

</Tab>
</Tabs>

## Optional Advanced Configuration

### Stream pools

A stream pool is useful only when an algorithm has independent work that can run concurrently. For example, an algorithm might search several independent shards or launch separate preprocessing work on different streams. Start with the default behavior, then configure a small stream pool only when benchmarks show it helps.

<Tabs>
<Tab title="C++">

```cpp
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <memory>

raft::device_resources resources;

auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(4);
raft::resource::set_cuda_stream_pool(resources, stream_pool);

// Algorithms that use the stream pool can now run independent work
// across up to four CUDA streams.
```

</Tab>
</Tabs>

### Workspace resources

Workspace resources control where algorithms allocate temporary buffers. The default behavior is usually fine for small examples, but production workloads can benefit from configuring workspace memory explicitly before arrays, indexes, or algorithm state are created.

NVIDIA cuVS uses two related workspace concepts:

- The workspace resource handles ordinary temporary allocations used during a computation.
- The large workspace resource handles bigger temporary allocations that should stay separate from the ordinary workspace pool.

For allocator choices such as device pools, pinned host memory, managed memory, and host memory, see the [Memory Management guide](/user-guide/api-guides/core-types/memory-management).

<Tabs>
<Tab title="C++">

```cpp
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/workspace_resource.hpp>

raft::device_resources resources;

// Give ordinary temporary allocations a bounded pool.
raft::resource::set_workspace_to_pool_resource(resources, 2 * 1024 * 1024 * 1024ull);

// Call cuVS algorithms after the workspace has been configured.
```

</Tab>
</Tabs>

The large workspace resource is the RAFT-side hook for what many applications treat as a big memory resource: memory reserved for especially large temporary buffers. This is useful in services or benchmarking harnesses that run workloads with very different temporary memory shapes.

<Tabs>
<Tab title="C++">

```cpp
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/workspace_resource.hpp>

raft::device_resources resources;

auto large_workspace = get_or_create_shared_large_workspace_resource();
raft::resource::set_large_workspace_resource(
    resources, raft::mr::device_resource{large_workspace});

// Run algorithms that may need large temporary allocations.
```

</Tab>
</Tabs>

## Important Resource Types

| Type | Where it appears | Purpose |
| --- | --- | --- |
| `raft::device_resources` | Single-GPU C++ APIs | The usual C++ resources object for one GPU. |
| `raft::resources` | Lower-level C++ RAFT and NVIDIA cuVS APIs | A resource container used by C++ algorithms and advanced applications. |
| `raft::memory_tracking_resources` | C++ diagnostics | A `raft::resources` handle that wraps all reachable memory resources and logs allocation samples to a CSV file from a background thread. See [Memory tracking](#memory-tracking). |
| `raft::memory_stats_resources` | C++ diagnostics | A `raft::resources` handle that wraps all reachable memory resources with statistics adaptors so peak/current byte usage and allocation counts can be queried in-process via `get_bytes_peak()`, `get_bytes_current()`, and related accessors. |
| `raft::device_resources_snmg` | Single-node multi-GPU C++ APIs | A convenience layer for one process controlling multiple GPUs. See [Multi-GPU](/user-guide/api-guides/core-types/multi-gpu). |
| `cuvsResources_t` | C API and language bindings | Opaque handle over RAFT resources for ABI-stable bindings. |
| `Resources` | Python and Rust | Language wrapper around `cuvsResources_t`. |
| `Resource` | Go | Go wrapper around `cuvsResources_t`. |
| `CuVSResources` | Java | Java `AutoCloseable` wrapper around native NVIDIA cuVS resources. |


### Memory tracking

A resources handle whose memory allocations are tracked and written as CSV samples from a background thread can be created in any of the supported languages.
The handle wraps all reachable memory resources (host, pinned, managed, device, workspace, large_workspace) with allocation-tracking adaptors and replaces the global host and device memory resources for the lifetime of the handle.
It is otherwise indistinguishable from a regular resources handle and can be passed to every NVIDIA cuVS API that accepts one.
The CSV reporter is stopped and the global memory resources are restored when the handle is destroyed.

<Note>

- The handle replaces the **global** host and device memory resources while it is alive.
  Do not create multiple tracking handles concurrently and make sure the handle outlives every consumer (matrices, indexes, search results, ...) that allocates memory through NVIDIA cuVS.
- The CSV file is flushed eagerly: the header is flushed on construction and every sample row is flushed as soon as it is written, so the file can be tailed while the handle is alive.
  This helps during debugging, if the program has segfaulted.
  Destroying the handle stops the background sampler and writes one final row.
- The sample interval is a *minimum* time between samples.
  The background thread blocks until an allocation/deallocation occurs, then sleeps for at least `sample_interval` before writing the next row; quiescent periods do not produce extra rows.

</Note>

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>

#include <cuda_runtime.h>

cuvsResources_t res;
// 10 ms sampling matches the C++ default.
cuvsResourcesCreateWithMemoryTracking(&res, "/tmp/allocations.csv", 10);

// ... do some processing ...

cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <raft/core/memory_tracking_resources.hpp>

// Sample interval defaults to std::chrono::milliseconds{10}.
raft::memory_tracking_resources res{"/tmp/allocations.csv"};

// ... do some processing ...
// `res` is implicitly convertible to raft::resources& and can be passed
// to any cuVS / RAFT API that accepts a resources handle.
```

</Tab>
<Tab title="Python">

```python
from cuvs.common import Resources

res = Resources(
    memory_tracking_csv_path="/tmp/allocations.csv",
    memory_tracking_sample_interval_ms=10,
)

# ... do some processing ...

del res  # flushes the CSV and restores the global memory resources
```

</Tab>
<Tab title="Java">

```java
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.nio.file.Path;
import java.time.Duration;

try (var res = CuVSResources.create(
        CuVSProvider.tempDirectory(),
        Path.of("/tmp/allocations.csv"),
        Duration.ofMillis(10))) {
    // ... do some processing ...
}
```

</Tab>
<Tab title="Rust">

```rust
use std::time::Duration;

let res = cuvs::Resources::with_memory_tracking(
    "/tmp/allocations.csv",
    Some(Duration::from_millis(10)),
)?;
```

</Tab>
</Tabs>

## Practical Guidance

Use the default resources behavior for simple one-off calls.

Create and reuse a resources object when a workflow has multiple related calls. Operations enqueued on the same resources object run in stream order, so you can chain work and synchronize once at the end.

Use one resources object per host thread. Resources are not thread-safe for concurrent use.

Synchronize explicitly before the host reads GPU results or before measuring runtime. Avoid synchronizing between every NVIDIA cuVS call unless the application actually needs the intermediate result on the host.

Configure memory resources, workspace resources, and stream pools before allocating inputs or building indexes. Changing resource configuration halfway through a workflow makes ownership and lifetime harder to reason about.

For multi-GPU workloads, choose the resource type first: `raft::device_resources_snmg` for single-node multi-GPU, or a RAFT resources object with an NCCL communicator for multi-node work. The [Multi-GPU guide](/user-guide/api-guides/core-types/multi-gpu) covers those setup patterns.
