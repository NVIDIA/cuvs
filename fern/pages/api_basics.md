# NVIDIA cuVS API Basics

- [Memory management](#memory-management)
- [Resource management](#resource-management)
- [Memory tracking](#memory-tracking)

## Memory management

Centralized memory management allows flexible configuration of allocation strategies, such as sharing the same CUDA memory pool across library boundaries. NVIDIA cuVS uses the [RMM](https://github.com/rapidsai/rmm) library, which eases the burden of configuring different allocation strategies globally across GPU-accelerated libraries.

RMM currently has APIs for C++ and Python.

### C++

Here's an example of configuring RMM to use a pool allocator in C++ (derived from the RMM example [here](https://github.com/rapidsai/rmm?tab=readme-ov-file#example)):

```c++
rmm::mr::cuda_memory_resource cuda_mr;
// Construct a resource that uses a coalescing best-fit pool allocator
// With the pool initially half of available device memory
auto initial_size = rmm::percent_of_free_device_memory(50);
rmm::mr::pool_memory_resource pool_mr{cuda_mr, initial_size};
rmm::mr::set_current_device_resource(pool_mr);
auto mr = rmm::mr::get_current_device_resource_ref();
```

### Python

And the corresponding code in Python (derived from the RMM example [here](https://github.com/rapidsai/rmm?tab=readme-ov-file#memoryresource-objects)):

```python
import rmm
pool = rmm.mr.PoolMemoryResource(
  rmm.mr.CudaMemoryResource(),
  initial_pool_size=2**30,
  maximum_pool_size=2**32)
rmm.mr.set_current_device_resource(pool)
```

## Resource management

NVIDIA cuVS uses an API from the [RAFT](https://github.com/rapidsai/raft) library of ML and data mining primitives to centralize and reuse expensive resources, such as memory management. The below code examples demonstrate how to create these resources for use throughout this guide.

See RAFT's [resource API documentation](https://docs.rapids.ai/api/raft/nightly/cpp_api/core_resources/) for more information.

C
^

```c
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>

cuvsResources_t res;
cuvsResourcesCreate(&res);

// ... do some processing ...

cuvsResourcesDestroy(res);
```

### C++

```c++
#include <raft/core/device_resources.hpp>

raft::device_resources res;
```

### Python

```python
import pylibraft

res = pylibraft.common.DeviceResources()
```

### Rust

```rust
let res = cuvs::Resources::new()?;
```

## Memory tracking

A resources handle whose memory allocations are tracked and written as CSV samples from a background thread can be created in any of the supported languages. The handle wraps all reachable memory resources (host, pinned, managed, device, workspace, large_workspace) with allocation-tracking adaptors and replaces the global host and device memory resources for the lifetime of the handle. It is otherwise indistinguishable from a regular resources handle and can be passed to every cuVS API that accepts one. The CSV reporter is stopped and the global memory resources are restored when the handle is destroyed.

<Note>

- The handle replaces the **global** host and device memory resources while it is alive. Do not create multiple tracking handles concurrently and make sure the handle outlives every consumer (matrices, indexes, search results, ...) that allocates memory through cuVS.
- The CSV file is flushed eagerly: the header is flushed on construction and every sample row is flushed as soon as it is written, so the file can be tailed while the handle is alive. Destroying the handle stops the background sampler and writes one final row.
- The sample interval is a *minimum* time between samples. The background thread blocks until an allocation/deallocation occurs, then sleeps for at least `sample_interval` before writing the next row; quiescent periods do not produce extra rows.

</Note>

### C

```c
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>

cuvsResources_t res;
// 10 ms sampling matches the C++ default.
cuvsResourcesCreateWithMemoryTracking(&res, "/tmp/allocations.csv", 10);

// ... do some processing ...

cuvsResourcesDestroy(res);
```

### C++

```c++
#include <raft/core/memory_tracking_resources.hpp>

// Sample interval defaults to std::chrono::milliseconds{10}.
raft::memory_tracking_resources res{"/tmp/allocations.csv"};

// ... do some processing ...
// `res` is implicitly convertible to raft::resources& and can be passed
// to any cuVS / raft API that accepts a resources handle.
```

### Python

```python
from cuvs.common import Resources

res = Resources(
    memory_tracking_csv_path="/tmp/allocations.csv",
    memory_tracking_sample_interval_ms=10,
)

# ... do some processing ...

del res  # flushes the CSV and restores the global memory resources
```

### Java

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

### Rust

```rust
use std::time::Duration;

let res = cuvs::Resources::with_memory_tracking(
    "/tmp/allocations.csv",
    Some(Duration::from_millis(10)),
)?;
```
