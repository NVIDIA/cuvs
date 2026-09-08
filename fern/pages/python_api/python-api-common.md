---
slug: api-reference/python-api-common
---

# Common

_Python module: `cuvs.common`_

## auto_sync_resources

```python
def auto_sync_resources(f)
```

Decorator to automatically call sync on a cuVS Resources object when
it isn't passed to a function.

When a resources=None is passed to the wrapped function, this decorator
will automatically create a default resources for the function, and
call sync on that resources when the function exits.

This will also insert the appropriate docstring for the resources parameter

## Resources

```python
cdef class Resources
```

Resources  is a lightweight python wrapper around the corresponding
C++ class of resources exposed by RAFT's C++ interface. Refer to
the header file raft/core/resources.hpp for interface level
details of this struct.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `stream` | `Optional stream to use for ordering CUDA instructions` |  |
| `memory_tracking_csv_path` | `Optional path-like` | If provided, the handle wraps all reachable memory resources (host, pinned, managed, device, workspace, large_workspace) with allocation-tracking adaptors and logs CSV samples to the given file from a background thread. The CSV file is created or truncated. The global host and device memory resources are replaced for the lifetime of the handle and restored when the handle is destroyed. |
| `memory_tracking_sample_interval_ms` | `int, default \`\`10\`\`` | Minimum interval between successive CSV samples, in milliseconds. Ignored when ``memory_tracking_csv_path`` is ``None``. |

**Examples**

Basic usage:

```python
>>> from cuvs.common import Resources
>>> handle = Resources()
>>>
>>> # call algos here
>>>
>>> # final sync of all work launched in the stream of this handle
>>> handle.sync()
```

Using a cuPy stream with cuVS Resources:

```python
>>> import cupy
>>> from cuvs.common import Resources
>>>
>>> cupy_stream = cupy.cuda.Stream()
>>> handle = Resources(stream=cupy_stream.ptr)
```

Tracking memory allocations to a CSV file:

```python
>>> from cuvs.common import Resources
>>> handle = Resources(memory_tracking_csv_path="/tmp/allocations.csv",
...                    memory_tracking_sample_interval_ms=10)  # doctest: +SKIP
```

**Members**

| Name | Kind |
| --- | --- |
| `sync` | method |
| `get_c_obj` | method |

### sync

```python
def sync(self)
```

### get_c_obj

```python
def get_c_obj(self)
```

Return the pointer to the underlying c_obj as a size_t

## MultiGpuResources

```python
cdef class MultiGpuResources
```

Multi-GPU Resources is a lightweight python wrapper around the
corresponding C++ class of multi-GPU resources exposed by RAFT's C++
interface. This class provides a handle for multi-GPU operations across
all available GPUs.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `stream` | `int, optional` | A CUDA stream pointer to use for this resource handle. If None, a<br />default stream will be used. |
| `device_ids` | `list of int, optional` | A list of device IDs to use for multi-GPU operations. If None, all available GPUs will be used. |

**Examples**

Basic usage:

```python
>>> from cuvs.common import MultiGpuResources
>>> handle = MultiGpuResources()
>>>
>>> # call multi-GPU algos here
>>>
>>> # final sync of all work launched in the stream of this handle
>>> handle.sync()
```

Using a cuPy stream with cuVS Multi-GPU Resources:

```python
>>> import cupy
>>> from cuvs.common import MultiGpuResources
>>>
>>> cupy_stream = cupy.cuda.Stream()
>>> handle = MultiGpuResources(stream=cupy_stream.ptr)
```

Using specific device IDs:

```python
>>> from cuvs.common import MultiGpuResources
>>> handle = MultiGpuResources(device_ids=[0])
>>>
>>> # call multi-GPU algos here
>>>
>>> handle.sync()
```

**Members**

| Name | Kind |
| --- | --- |
| `sync` | method |
| `set_memory_pool` | method |
| `get_c_obj` | method |

### sync

```python
def sync(self)
```

### set_memory_pool

```python
def set_memory_pool(self, percent_of_free_memory)
```

Set a memory pool on all devices managed by these resources.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `percent_of_free_memory` | `int` | Percentage of free device memory to allocate for the pool. |

**Examples**

```python
>>> from cuvs.common import MultiGpuResources
>>> handle = MultiGpuResources()
>>> handle.set_memory_pool(80)  # Use 80% of free memory
```

### get_c_obj

```python
def get_c_obj(self)
```

Return the pointer to the underlying c_obj as a size_t

## auto_sync_multi_gpu_resources

```python
def auto_sync_multi_gpu_resources(f)
```

Decorator to automatically call sync on a cuVS Multi-GPU Resources
object when it isn't passed to a function.

When a resources=None is passed to the wrapped function, this decorator
will automatically create a default multi-GPU resources for the function,
and call sync on that resources when the function exits.

This will also insert the appropriate docstring for the resources
parameter

## Dataset

```python
cdef class Dataset
```

Wrapper around a ``cuvsDataset`` handle.

**Members**

| Name | Kind |
| --- | --- |
| `memory_type` | property |
| `layout` | property |
| `is_owning` | property |
| `dtype` | property |

### memory_type

```python
def memory_type(self)
```

### layout

```python
def layout(self)
```

### is_owning

```python
def is_owning(self)
```

### dtype

```python
def dtype(self)
```

## make_device_padded_dataset

`@auto_sync_resources`

```python
def make_device_padded_dataset(dataset, resources=None)
```

Create a device-padded ``Dataset`` from a host or device array.

The input must be a row-major 2-D matrix. Host arrays are always copied into
newly allocated device-padded storage (``is_owning`` is ``True``). Device
arrays are copied when their row stride does not already match the
required padded width; if the stride is already correct, a non-owning
padded view of the input is returned and the caller must keep ``dataset``
alive for as long as the ``Dataset`` is used.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dataset` | `array interface compliant matrix, shape \`\`(n_samples, dim)\`\`` | Host (e.g. NumPy) or device (e.g. CuPy) array. Supported dtypes are ``float32``, ``float16``, ``int8``, and ``uint8``. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `dataset` | `Dataset` | A device-resident padded dataset handle. Check ``is_owning`` to see whether the handle owns its storage or is a view of ``dataset``. |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.common import make_device_padded_dataset
>>> X = cp.random.random_sample((1000, 50), dtype=cp.float32)
>>> ds = make_device_padded_dataset(X)
>>> ds.memory_type
'device'
>>> ds.layout
'padded'
```
