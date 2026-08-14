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
