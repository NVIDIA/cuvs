---
slug: api-reference/cpp-api-common-types-execution-resources
---

# Execution Resources

Most NVIDIA cuVS C++ APIs accept `raft::resources const&` as the first argument. Resource objects carry execution state such as CUDA streams, CUDA library handles, memory resources, stream pools, and communication resources.

<a id="raft-resources"></a>
### raft::resources

_Source header: `raft/core/resources.hpp`_

Primary execution context passed to most NVIDIA cuVS C++ APIs. The object gives algorithms access to shared CUDA streams, library handles, memory resources, and other lazily-created state.

```cpp
class resources;
```

<a id="raft-resource-get-cuda-stream"></a>
#### raft::resource::get_cuda_stream

_Source header: `raft/core/resource/cuda_stream.hpp`_

Returns the CUDA stream associated with a resources object.

```cpp
rmm::cuda_stream_view get_cuda_stream(raft::resources const& res);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object to query. |

**Returns**

`rmm::cuda_stream_view`

<a id="raft-resource-sync-stream"></a>
#### raft::resource::sync_stream

_Source header: `raft/core/resource/cuda_stream.hpp`_

Synchronizes the CUDA stream associated with a resources object.

```cpp
void sync_stream(raft::resources const& res);
void sync_stream(raft::resources const& res, rmm::cuda_stream_view stream);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object to synchronize. |
| `stream` | `rmm::cuda_stream_view` | Optional stream to synchronize instead of the main stream. |

**Returns**

`void`

<a id="raft-resource-set-cuda-stream-pool"></a>
#### raft::resource::set_cuda_stream_pool

_Source header: `raft/core/resource/cuda_stream_pool.hpp`_

Attaches a CUDA stream pool to a resources object so algorithms can issue independent work on multiple streams.

```cpp
void set_cuda_stream_pool(raft::resources const& res,
                          std::shared_ptr<rmm::cuda_stream_pool> pool);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object to configure. |
| `pool` | `std::shared_ptr<rmm::cuda_stream_pool>` | Stream pool to attach. |

**Returns**

`void`

<a id="raft-resource-get-stream-from-stream-pool"></a>
#### raft::resource::get_stream_from_stream_pool

_Source header: `raft/core/resource/cuda_stream_pool.hpp`_

Returns a stream from the configured stream pool.

```cpp
rmm::cuda_stream_view get_stream_from_stream_pool(raft::resources const& res);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object to query. |

**Returns**

`rmm::cuda_stream_view`

<a id="raft-resource-sync-stream-pool"></a>
#### raft::resource::sync_stream_pool

_Source header: `raft/core/resource/cuda_stream_pool.hpp`_

Synchronizes streams in the configured stream pool.

```cpp
void sync_stream_pool(raft::resources const& res);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object to synchronize. |

**Returns**

`void`

<a id="raft-resource-set-workspace-to-pool-resource"></a>
#### raft::resource::set_workspace_to_pool_resource

_Source header: `raft/core/resource/workspace_resource.hpp`_

Configures workspace allocation for algorithms that need temporary device memory.

```cpp
void set_workspace_to_pool_resource(
  raft::resources const& res,
  std::optional<std::size_t> allocation_limit = std::nullopt);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object to configure. |
| `allocation_limit` | `std::optional<std::size_t>` | Optional temporary workspace allocation limit in bytes. |

**Returns**

`void`

<a id="raft-resource-comms-initialized"></a>
#### raft::resource::comms_initialized

_Source header: `raft/core/resource/comms.hpp`_

Reports whether communication resources have been initialized.

```cpp
bool comms_initialized(raft::resources const& res);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object to query. |

**Returns**

`bool`

<a id="raft-resource-is-multi-gpu"></a>
#### raft::resource::is_multi_gpu

_Source header: `raft/core/resource/comms.hpp`_

Reports whether a resources object is configured for multi-GPU use.

```cpp
bool is_multi_gpu(raft::resources const& res);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object to query. |

**Returns**

`bool`

<a id="raft-device-resources"></a>
### raft::device_resources

_Source header: `raft/core/device_resources.hpp`_

Convenience `raft::resources` implementation for single-GPU applications and examples.

```cpp
class device_resources;
```

<a id="raft-device-resources-device-resources"></a>
#### raft::device_resources::device_resources

Constructs a single-GPU resources object.

```cpp
device_resources(
  rmm::cuda_stream_view stream_view = rmm::cuda_stream_per_thread,
  std::shared_ptr<rmm::cuda_stream_pool> stream_pool = nullptr,
  std::shared_ptr<rmm::mr::device_memory_resource> workspace_resource = nullptr,
  std::optional<std::size_t> allocation_limit = std::nullopt);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `stream_view` | `rmm::cuda_stream_view` | Default CUDA stream used by algorithms. |
| `stream_pool` | `std::shared_ptr<rmm::cuda_stream_pool>` | Optional CUDA stream pool. |
| `workspace_resource` | `std::shared_ptr<rmm::mr::device_memory_resource>` | Optional workspace memory resource. |
| `allocation_limit` | `std::optional<std::size_t>` | Optional temporary workspace allocation limit in bytes. |

<a id="raft-device-resources-sync-stream"></a>
#### raft::device_resources::sync_stream

Synchronizes either the main stream or a specific CUDA stream.

```cpp
void sync_stream() const;
void sync_stream(rmm::cuda_stream_view stream) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `stream` | `rmm::cuda_stream_view` | Stream to synchronize. Omit to synchronize the main stream. |

**Returns**

`void`

<a id="raft-device-resources-get-stream"></a>
#### raft::device_resources::get_stream

Returns the main CUDA stream associated with the resources object.

```cpp
rmm::cuda_stream_view get_stream() const;
```

**Returns**

`rmm::cuda_stream_view`

<a id="raft-device-resources-is-stream-pool-initialized"></a>
#### raft::device_resources::is_stream_pool_initialized

Reports whether a CUDA stream pool is configured.

```cpp
bool is_stream_pool_initialized() const;
```

**Returns**

`bool`

<a id="raft-device-resources-get-stream-pool"></a>
#### raft::device_resources::get_stream_pool

Returns the configured CUDA stream pool.

```cpp
rmm::cuda_stream_pool const& get_stream_pool() const;
```

**Returns**

`rmm::cuda_stream_pool const&`

<a id="raft-device-resources-get-stream-from-stream-pool"></a>
#### raft::device_resources::get_stream_from_stream_pool

Returns a stream from the configured CUDA stream pool.

```cpp
rmm::cuda_stream_view get_stream_from_stream_pool() const;
rmm::cuda_stream_view get_stream_from_stream_pool(std::size_t stream_idx) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `stream_idx` | `std::size_t` | Optional index of the stream in the stream pool. |

**Returns**

`rmm::cuda_stream_view`

<a id="raft-device-resources-get-next-usable-stream"></a>
#### raft::device_resources::get_next_usable_stream

Returns a stream from the pool when one exists; otherwise returns the main stream.

```cpp
rmm::cuda_stream_view get_next_usable_stream() const;
rmm::cuda_stream_view get_next_usable_stream(std::size_t stream_idx) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `stream_idx` | `std::size_t` | Optional stream pool index to use when a stream pool is configured. |

**Returns**

`rmm::cuda_stream_view`

<a id="raft-device-resources-sync-stream-pool"></a>
#### raft::device_resources::sync_stream_pool

Synchronizes all streams in the pool or a subset of stream indices.

```cpp
void sync_stream_pool() const;
void sync_stream_pool(std::vector<std::size_t> stream_indices) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `stream_indices` | `std::vector<std::size_t>` | Optional stream indices to synchronize. |

**Returns**

`void`

<a id="raft-device-resources-wait-stream-pool-on-stream"></a>
#### raft::device_resources::wait_stream_pool_on_stream

Makes the stream pool wait on work submitted to the main stream.

```cpp
void wait_stream_pool_on_stream() const;
```

**Returns**

`void`

<a id="raft-device-resources-snmg"></a>
### raft::device_resources_snmg

_Source header: `raft/core/device_resources_snmg.hpp`_

Single-node multi-GPU resources object used by C++ APIs that operate over more than one local GPU.

```cpp
class device_resources_snmg;
```

<a id="raft-device-resources-snmg-device-resources-snmg"></a>
#### raft::device_resources_snmg::device_resources_snmg

Constructs single-node multi-GPU resources for all GPUs or a subset.

```cpp
device_resources_snmg();
device_resources_snmg(std::vector<int> const& device_ids);
device_resources_snmg(device_resources_snmg const& world);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `device_ids` | `std::vector<int> const&` | Optional list of local GPU device IDs to use. |
| `world` | `device_resources_snmg const&` | Existing single-node multi-GPU resources object to copy. |

<a id="raft-device-resources-snmg-set-memory-pool"></a>
#### raft::device_resources_snmg::set_memory_pool

Configures a memory pool on all GPUs managed by the resources object.

```cpp
void set_memory_pool(int percent_of_free_memory);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `percent_of_free_memory` | `int` | Percentage of free memory to reserve for each memory pool. |

**Returns**

`void`

<a id="raft-device-resources-snmg-has-resource-factory"></a>
#### raft::device_resources_snmg::has_resource_factory

Reports whether a resource factory is registered for a resource type.

```cpp
bool has_resource_factory(raft::resource::resource_type resource_type) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `resource_type` | `raft::resource::resource_type` | Resource type to check. |

**Returns**

`bool`
