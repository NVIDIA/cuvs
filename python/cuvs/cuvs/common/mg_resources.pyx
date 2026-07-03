#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuda.bindings.cyruntime cimport cudaStream_t

from cuvs.common.c_api cimport (
    cuvsMultiGpuResourcesCreate,
    cuvsMultiGpuResourcesCreateWithDeviceIds,
    cuvsMultiGpuResourcesDestroy,
    cuvsMultiGpuResourcesSetMemoryPool,
    cuvsResources_t,
    cuvsStreamSet,
    cuvsStreamSync,
)
from cuvs.common.cydlpack cimport DLManagedTensor, dlpack_c

import numpy as np
from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.exceptions import check_cuvs


cdef class MultiGpuResources:
    """
    Multi-GPU Resources is a lightweight python wrapper around the
    corresponding C++ class of multi-GPU resources exposed by RAFT's C++
    interface. This class provides a handle for multi-GPU operations across
    all available GPUs.

    Parameters
    ----------
    stream : int, optional
        A CUDA stream pointer to use for this resource handle. If None, a
        default stream will be used.
    device_ids : list of int, optional
        A list of device IDs to use for multi-GPU operations. If None, all
        available GPUs will be used.

    Examples
    --------

    Basic usage:

    >>> from cuvs.common import MultiGpuResources
    >>> handle = MultiGpuResources()
    >>>
    >>> # call multi-GPU algos here
    >>>
    >>> # final sync of all work launched in the stream of this handle
    >>> handle.sync()

    Using a cuPy stream with cuVS Multi-GPU Resources:

    >>> import cupy
    >>> from cuvs.common import MultiGpuResources
    >>>
    >>> cupy_stream = cupy.cuda.Stream()
    >>> handle = MultiGpuResources(stream=cupy_stream.ptr)

    Using specific device IDs:

    >>> from cuvs.common import MultiGpuResources
    >>> handle = MultiGpuResources(device_ids=[0])
    >>>
    >>> # call multi-GPU algos here
    >>>
    >>> handle.sync()
    """

    def __cinit__(self, stream=None, device_ids=None, handle=None):
        cdef DLManagedTensor* device_ids_dlpack = NULL

        if handle is not None:
            # Use existing handle (e.g., from pylibraft)
            self.c_obj = <cuvsResources_t><size_t>handle
            self._owns_resource = False
        elif device_ids is not None:
            # Convert device_ids list to DLManagedTensor (keep on host/CPU)
            # NumPy arrays are naturally on the host, which is what we need
            device_ids_array = np.asarray(device_ids, dtype=np.int32)
            ai = wrap_array(device_ids_array)
            device_ids_dlpack = dlpack_c(ai)
            check_cuvs(cuvsMultiGpuResourcesCreateWithDeviceIds(
                &self.c_obj, device_ids_dlpack))
            self._owns_resource = True
        else:
            check_cuvs(cuvsMultiGpuResourcesCreate(&self.c_obj))
            self._owns_resource = True

        if stream:
            check_cuvs(cuvsStreamSet(self.c_obj, <cudaStream_t>stream))

    def sync(self):
        check_cuvs(cuvsStreamSync(self.c_obj))

    def set_memory_pool(self, percent_of_free_memory):
        """
        Set a memory pool on all devices managed by these resources.

        Parameters
        ----------
        percent_of_free_memory : int
            Percentage of free device memory to allocate for the pool.

        Examples
        --------
        >>> from cuvs.common import MultiGpuResources
        >>> handle = MultiGpuResources()
        >>> handle.set_memory_pool(80)  # Use 80% of free memory
        """
        check_cuvs(cuvsMultiGpuResourcesSetMemoryPool(
            self.c_obj, percent_of_free_memory))

    def get_c_obj(self):
        """
        Return the pointer to the underlying c_obj as a size_t
        """
        return <size_t> self.c_obj

    def __dealloc__(self):
        if self._owns_resource:
            check_cuvs(cuvsMultiGpuResourcesDestroy(self.c_obj))


_multi_gpu_resources_param_string = """
     resources : Optional cuVS Multi-GPU Resource handle for reusing CUDA \
resources.
        If Multi-GPU Resources aren't supplied, CUDA resources will be
        allocated inside this function and synchronized before the
        function exits. If resources are supplied, you will need to
        explicitly synchronize yourself by calling `resources.sync()`
        before accessing the output. Also accepts pylibraft
        DeviceResourcesSNMG objects.
""".strip()


def auto_sync_multi_gpu_resources(f):
    """Decorator to automatically call sync on a cuVS Multi-GPU Resources
    object when it isn't passed to a function.

    When a resources=None is passed to the wrapped function, this decorator
    will automatically create a default multi-GPU resources for the function,
    and call sync on that resources when the function exits.

    This also accepts pylibraft DeviceResourcesSNMG objects, which will
    be wrapped to provide the cuVS MultiGpuResources interface.

    This will also insert the appropriate docstring for the resources
    parameter
    """
    import functools

    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        sync_resources = resources is None

        if resources is None:
            resources = MultiGpuResources()
        elif not isinstance(resources, MultiGpuResources):
            # Create a MultiGpuResources instance from pylibraft
            # DeviceResourcesSNMG
            if not hasattr(resources, 'getHandle'):
                raise TypeError(
                    "resources must be a cuVS MultiGpuResources or pylibraft "
                    "DeviceResourcesSNMG object"
                )
            resources = MultiGpuResources(handle=resources.getHandle())

        ret_value = f(*args, resources=resources, **kwargs)

        if sync_resources:
            resources.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        resources_docstring=_multi_gpu_resources_param_string
    )
    return wrapper
