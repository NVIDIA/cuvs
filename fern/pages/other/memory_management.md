# Memory Management

NVIDIA cuVS uses RAPIDS Memory Manager (RMM) through RAFT so GPU algorithms can allocate temporary buffers, output arrays, and staging memory through one configurable memory layer. RMM helps NVIDIA cuVS interoperate with the broader GPU library ecosystem that either uses RMM directly or provides RMM adaptors, including RAPIDS libraries, PyTorch, CuPy, Faiss, and TensorFlow. This lets applications share memory resources and memory allocations across library boundaries without unnecessary copies.

The most common choice is to configure a device memory pool before creating NVIDIA cuVS resources or allocating device arrays. Pooling avoids repeated `cudaMalloc` and `cudaFree` calls, which can synchronize the device and add allocator overhead to workloads with many temporary buffers.

## Example API Usage

[C resources API](/api-reference/c-api-core-c-api#cuvsrmmpoolmemoryresourceenable) | [Java provider API](/api-reference/java-api-com-nvidia-cuvs-spi-cuvsprovider#enablermmpooledmemory) | [Go memory API](/api-reference/go-api-cuvs#newcuvspoolmemory)

C, Java, Go, and Rust configure RMM through NVIDIA cuVS wrappers over the C API. C++ and Python applications usually use RMM directly because RMM is part of the RAPIDS stack used by RAFT and NVIDIA cuVS.

### Setting a device pool

Use a pool memory resource when a workload repeatedly builds indexes, searches in batches, runs clustering iterations, or allocates many temporary buffers. Configure the pool early, then keep it alive until all allocations that use it are complete.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>

int main(void)
{
  cuvsError_t error;

  error = cuvsRMMPoolMemoryResourceEnable(50, 90, false);
  if (error != CUVS_SUCCESS) { return 1; }

  // Create cuvsResources_t objects and call cuVS APIs here.

  error = cuvsRMMMemoryResourceReset();
  if (error != CUVS_SUCCESS) { return 1; }

  return 0;
}
```

</Tab>
<Tab title="C++">

```cpp
#include <rmm/mr/pool_memory_resource.hpp>

#include <raft/core/resources.hpp>

int main()
{
  raft::device_resources resources;

  auto initial_pool_size = 1024 * 1024 * 1024ull;
  rmm::mr::pool_memory_resource pool_mr(
      rmm::mr::get_current_device_resource_ref(), initial_pool_size);

  auto previous_mr = rmm::mr::set_current_device_resource(pool_mr);

  // Allocate inputs and call cuVS APIs here.

  rmm::mr::set_current_device_resource(previous_mr);
  return 0;
}
```

</Tab>
<Tab title="Python">

```python
import rmm

pool = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),
    initial_pool_size=1 << 30,
    maximum_pool_size=8 << 30,
)
rmm.mr.set_current_device_resource(pool)

# Allocate CuPy, cuDF, or cuVS inputs and call cuVS APIs here.
```

</Tab>
<Tab title="Java">

```java
import com.nvidia.cuvs.spi.CuVSProvider;

public class MemoryPoolExample {
  public static void main(String[] args) {
    CuVSProvider.provider().enableRMMPooledMemory(50, 90);

    try {
      // Create CuVSResources, matrices, indexes, and run cuVS APIs here.
    } finally {
      CuVSProvider.provider().resetRMMPooledMemory();
    }
  }
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs_sys::{
    cuvsError_t, cuvsRMMMemoryResourceReset, cuvsRMMPoolMemoryResourceEnable,
};

fn check(status: cuvsError_t) -> Result<(), &'static str> {
    match status {
        cuvsError_t::CUVS_SUCCESS => Ok(()),
        _ => Err("cuVS RMM call failed"),
    }
}

fn main() -> Result<(), &'static str> {
    unsafe {
        check(cuvsRMMPoolMemoryResourceEnable(50, 90, false))?;
    }

    // Create cuvs::Resources and call cuVS Rust APIs here.

    unsafe {
        check(cuvsRMMMemoryResourceReset())?;
    }

    Ok(())
}
```

</Tab>
<Tab title="Go">

```go
package main

import cuvs "github.com/rapidsai/cuvs/go"

func main() error {
	mem, err := cuvs.NewCuvsPoolMemory(50, 90, false)
	if err != nil {
		return err
	}
	defer mem.Close()

	// Create resources and call cuVS Go APIs here.
	return nil
}
```

</Tab>
</Tabs>

The C, Java, Rust, and Go examples change the current device resource through the NVIDIA cuVS C API. This has process-wide effect for the current device, so configure it before allocating long-lived objects and reset it only after those objects are destroyed.

### Allocating device memory

Use device memory for GPU-resident inputs, outputs, indexes, and scratch buffers. The default RMM device resource allocates CUDA device memory directly. A pool resource can sit above that default resource and serve the same allocations from a cached block of memory.

For examples of passing dense arrays into NVIDIA cuVS APIs across different languages and libraries, see [Using dense arrays in cuVS APIs](/user-guide/api-guides/core-types/array-types/dense-arrays#using-dense-arrays-in-cuvs-ap-is) in the Dense Arrays guide.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>

#include <stddef.h>

void allocate_device_buffer(size_t bytes)
{
  cuvsResources_t res;
  void* buffer = NULL;

  cuvsResourcesCreate(&res);

  cuvsRMMAlloc(res, &buffer, bytes);

  // Use buffer as a device allocation.

  cuvsRMMFree(res, buffer, bytes);
  cuvsResourcesDestroy(res);
}
```

</Tab>
<Tab title="C++">

```cpp
#include <rmm/device_uvector.hpp>

#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cstddef>

void allocate_device_buffer(raft::device_resources const& resources,
                            std::size_t bytes)
{
  auto stream = raft::resource::get_cuda_stream(resources);

  rmm::device_uvector<char> buffer(bytes, stream);

  // Use buffer.data() as a device allocation.
}
```

</Tab>
<Tab title="Python">

```python
import rmm

buffer = rmm.DeviceBuffer(size=1 << 20)

# Pass objects backed by RMM-allocated device memory to GPU libraries.
```

</Tab>
</Tabs>

Most users do not allocate raw buffers directly. NVIDIA cuVS APIs typically accept matrices, tensors, or language-native array wrappers, and those objects allocate through the active RMM resource underneath.

### Allocating pinned host memory

Pinned host memory is page-locked CPU memory. It is useful when data must be copied between CPU and GPU asynchronously, or when GPU kernels and CPU code need fast shared access to small host-side coordination buffers.

Use it selectively. Pinned memory is a limited system resource, and overusing it can reduce host-memory flexibility for the operating system.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>

#include <stddef.h>

void allocate_pinned_buffer(size_t bytes)
{
  void* host_buffer = NULL;

  cuvsRMMHostAlloc(&host_buffer, bytes);

  // Use host_buffer for pinned host staging.

  cuvsRMMHostFree(host_buffer, bytes);
}
```

</Tab>
<Tab title="C++">

```cpp
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cstddef>

void allocate_pinned_buffer(std::size_t bytes)
{
  rmm::mr::pinned_host_memory_resource pinned_mr;

  void* host_buffer = pinned_mr.allocate_sync(bytes);

  // Use host_buffer for pinned host staging.

  pinned_mr.deallocate_sync(host_buffer, bytes);
}
```

</Tab>
<Tab title="Python">

```python
import ctypes

import cupy as cp
import numpy as np
import rmm

n = 1 << 20
dtype = np.dtype(np.float32)
nbytes = n * dtype.itemsize

pinned_mr = rmm.mr.PinnedHostMemoryResource()
ptr = pinned_mr.allocate(nbytes)

try:
    host_type = ctypes.c_float * n
    host = np.ctypeslib.as_array(host_type.from_address(ptr))
    host[:] = np.arange(n, dtype=dtype)

    device = cp.empty(n, dtype=cp.float32)
    device.set(host)
finally:
    pinned_mr.deallocate(ptr, nbytes)
```

</Tab>
</Tabs>

Java NVIDIA cuVS resources use pinned host buffers internally for batched transfers. Most Java users should rely on `CuVSResources` and matrix builders instead of managing pinned memory directly.

### Using managed memory

Managed memory can simplify workflows where data may move between CPU and GPU address spaces or where an application wants a larger unified allocation model. It is useful for prototyping and some oversubscription workflows, but it can introduce page migration overhead. For performance-sensitive paths, benchmark managed memory against device memory.

<Tabs>
<Tab title="C">

```c
#include <cuvs/core/c_api.h>

int main(void)
{
  cuvsRMMPoolMemoryResourceEnable(50, 90, true);

  // cuVS allocations now use a managed-memory-backed pool.

  cuvsRMMMemoryResourceReset();
  return 0;
}
```

</Tab>
<Tab title="C++">

```cpp
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

void configure_managed_pool()
{
  rmm::mr::managed_memory_resource managed_mr;
  rmm::mr::pool_memory_resource pool_mr(
      managed_mr, 1024 * 1024 * 1024ull, 8 * 1024 * 1024 * 1024ull);

  auto previous_mr = rmm::mr::set_current_device_resource(pool_mr);

  // Allocate inputs and call cuVS APIs here.

  rmm::mr::set_current_device_resource(previous_mr);
}
```

</Tab>
<Tab title="Python">

```python
import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

pool = rmm.mr.PoolMemoryResource(
    rmm.mr.ManagedMemoryResource(),
    initial_pool_size=1 << 30,
    maximum_pool_size=8 << 30,
)
rmm.mr.set_current_device_resource(pool)
cp.cuda.set_allocator(rmm_cupy_allocator)

# CuPy arrays now allocate through the RMM managed-memory-backed pool.
data = cp.empty((1 << 20,), dtype=cp.float32)
data.fill(1.0)
```

</Tab>
<Tab title="Java">

```java
import com.nvidia.cuvs.spi.CuVSProvider;

CuVSProvider.provider().enableRMMManagedPooledMemory(50, 90);
try {
  // Create cuVS objects and run APIs here.
} finally {
  CuVSProvider.provider().resetRMMPooledMemory();
}
```

</Tab>
<Tab title="Rust">

```rust
use cuvs_sys::{
    cuvsError_t, cuvsRMMMemoryResourceReset, cuvsRMMPoolMemoryResourceEnable,
};

fn check(status: cuvsError_t) -> Result<(), &'static str> {
    match status {
        cuvsError_t::CUVS_SUCCESS => Ok(()),
        _ => Err("cuVS RMM call failed"),
    }
}

fn main() -> Result<(), &'static str> {
    unsafe {
        check(cuvsRMMPoolMemoryResourceEnable(50, 90, true))?;
    }

    // cuVS Rust APIs now allocate through a managed-memory-backed pool.

    unsafe {
        check(cuvsRMMMemoryResourceReset())?;
    }

    Ok(())
}
```

</Tab>
<Tab title="Go">

```go
package main

import cuvs "github.com/rapidsai/cuvs/go"

func main() error {
	mem, err := cuvs.NewCuvsPoolMemory(50, 90, true)
	if err != nil {
		return err
	}
	defer mem.Close()

	// cuVS Go APIs now allocate through a managed-memory-backed pool.
	return nil
}
```

</Tab>
</Tabs>

## How Memory Management works

RMM separates allocation policy from algorithm code. An NVIDIA cuVS algorithm asks RAFT for memory through the active resource, and the resource decides whether the allocation comes from direct device memory, a pool, managed memory, or a host allocation path.

This design gives users four practical benefits:

- Allocation behavior can be tuned without changing NVIDIA cuVS API calls.
- NVIDIA cuVS can share memory resources and allocations with other RMM-aware libraries, including RAPIDS libraries, PyTorch, CuPy, Faiss, and TensorFlow.
- Data can move through compatible GPU libraries without extra copies solely to cross library boundaries.
- Temporary allocations can be pooled to reduce allocator synchronization and fragmentation.

The active device resource should be configured before creating long-lived arrays, indexes, or NVIDIA cuVS resources. Changing the resource while live allocations still exist can make ownership hard to reason about, especially in applications that use several GPU libraries at once.

## When to use each resource

| Resource | Use it when | Notes |
| --- | --- | --- |
| Device memory resource | You want ordinary CUDA device allocations and a simple baseline. | This is the default allocation path. It is simple, but repeated direct allocations can be expensive in iterative workloads. |
| Pool memory resource | The workload performs many allocations or uses NVIDIA cuVS repeatedly in one process. | Usually the best first choice for production and benchmarking. Set the pool once near process startup. |
| Pinned host memory resource | CPU buffers are used for asynchronous copies or fast CPU/GPU staging. | Use for transfer staging and coordination buffers, not as a replacement for ordinary host memory. |
| Host memory resource | Data stays CPU-resident and does not need page-locked transfer behavior. | Regular host allocations are appropriate for input preparation, metadata, and CPU-side results. |
| Managed memory resource | You need unified addressing, simpler ownership, or controlled oversubscription. | Easier to use in some workflows, but page migration can hurt latency and throughput. Benchmark before using it for hot paths. |

## Configuration choices

Start with a device pool when a workload is allocation-heavy. For C, Java, Rust, and Go wrappers, the initial and maximum pool sizes are expressed as percentages of free device memory at configuration time.

Larger initial pools reduce the chance of later upstream allocations, but reserve more memory immediately. Larger maximum pools allow more growth, but can compete with other applications or GPU libraries on the same device.

For multi-GPU workloads, configure memory on each participating GPU before running the algorithm. See the [Multi-GPU guide](/user-guide/api-guides/core-types/multi-gpu) for resource initialization patterns.

## Practical guidance

Use a pool for repeatable performance measurements. Allocator behavior can otherwise appear as noise in benchmark results.

Keep the memory resource alive for at least as long as the allocations that use it. In C++ this means the pool object must outlive arrays allocated from it. In C, Java, Rust, and Go this means resetting the NVIDIA cuVS memory resource after NVIDIA cuVS objects have been destroyed.

Avoid changing memory resources in the middle of a workflow unless the application has a clear ownership boundary. The safest pattern is to configure once, allocate and run NVIDIA cuVS work, destroy NVIDIA cuVS objects, then reset.
