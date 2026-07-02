# NVIDIA cuVS API Basics

- [Memory management](#memory-management)
- [Resource management](#resource-management)

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
