#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import functools

from libc.stdint cimport uintptr_t

from cuvs.common.c_api cimport (
    cudaStream_t,
    cuvsResources_t,
    cuvsResourcesCreate,
    cuvsResourcesDestroy,
    cuvsStreamSet,
    cuvsStreamSync,
)

from cuvs.common.exceptions import check_cuvs


cdef class Resources:
    """
    Resources  is a lightweight python wrapper around the corresponding
    C++ class of resources exposed by RAFT's C++ interface. Refer to
    the header file raft/core/resources.hpp for interface level
    details of this struct.

    Parameters
    ----------
    stream : Optional stream to use for ordering CUDA instructions

    Examples
    --------

    Basic usage:

    >>> from cuvs.common import Resources
    >>> handle = Resources()
    >>>
    >>> # call algos here
    >>>
    >>> # final sync of all work launched in the stream of this handle
    >>> handle.sync()

    Using a cuPy stream with cuVS Resources:

    >>> import cupy
    >>> from cuvs.common import Resources
    >>>
    >>> cupy_stream = cupy.cuda.Stream()
    >>> handle = Resources(stream=cupy_stream.ptr)
    """

    def __cinit__(self, stream=None):
        check_cuvs(cuvsResourcesCreate(&self.c_obj))
        if stream is not None:
            check_cuvs(cuvsStreamSet(self.c_obj, <cudaStream_t><uintptr_t>stream))

    def sync(self):
        check_cuvs(cuvsStreamSync(self.c_obj))

    def get_c_obj(self):
        """
        Return the pointer to the underlying c_obj as a size_t
        """
        return <size_t> self.c_obj

    def __dealloc__(self):
        check_cuvs(cuvsResourcesDestroy(self.c_obj))


_resources_param_string = """
     resources : Optional cuVS Resource handle for reusing CUDA resources.
        If Resources aren't supplied, CUDA resources will be
        allocated inside this function and synchronized before the
        function exits. If resources are supplied, you will need to
        explicitly synchronize yourself by calling `resources.sync()`
        before accessing the output.
""".strip()


def auto_sync_resources(f):
    """Decorator to automatically call sync on a cuVS Resources object when
    it isn't passed to a function.

    When a resources=None is passed to the wrapped function, this decorator
    will automatically create a default resources for the function, and
    call sync on that resources when the function exits.

    This will also insert the appropriate docstring for the resources parameter
    """

    @functools.wraps(f)
    def wrapper(*args, resources=None, **kwargs):
        sync_resources = resources is None
        resources = resources if resources is not None else Resources()

        ret_value = f(*args, resources=resources, **kwargs)

        if sync_resources:
            resources.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        resources_docstring=_resources_param_string
    )
    return wrapper
