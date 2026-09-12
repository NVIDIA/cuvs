#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

from cuda.bindings.cyruntime cimport (
    cudaError,
    cudaError_t,
    cudaPointerAttributes,
    cudaPointerGetAttributes,
)
from libc cimport stdlib
from libc.stdint cimport uintptr_t


cdef void deleter(DLManagedTensor* tensor) noexcept:
    if tensor.manager_ctx is NULL:
        return
    stdlib.free(tensor.dl_tensor.shape)
    if tensor.dl_tensor.strides is not NULL:
        stdlib.free(tensor.dl_tensor.strides)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)


def dl_data_type_to_numpy(DLDataType dtype):
    """ Converts a DLDataType dtype to a numpy dtype """
    if dtype.code == DLDataTypeCode.kDLFloat:
        if dtype.bits == 32:
            return np.float32
        elif dtype.bits == 64:
            return np.float64
        elif dtype.bits == 16:
            return np.float16
        else:
            raise ValueError(f"unknown float dtype bits: {dtype.bits}")
    elif dtype.code == DLDataTypeCode.kDLInt:
        if dtype.bits == 32:
            return np.int32
        elif dtype.bits == 64:
            return np.int64
        elif dtype.bits == 16:
            return np.int16
        elif dtype.bits == 8:
            return np.int8
        else:
            raise ValueError(f"unknown int dtype bits: {dtype.bits}")
    elif dtype.code == DLDataTypeCode.kDLUInt:
        if dtype.bits == 32:
            return np.uint32
        elif dtype.bits == 64:
            return np.uint64
        elif dtype.bits == 16:
            return np.uint16
        elif dtype.bits == 8:
            return np.uint8
        else:
            raise ValueError(f"unknown uint dtype bits: {dtype.bits}")
    else:
        raise ValueError(f"unknown DLDataTypeCode.code: {dtype.code}")


cdef int _cuda_pointer_device_id(uintptr_t tensor_ptr) except *:
    cdef cudaPointerAttributes attributes
    cdef cudaError_t status = cudaPointerGetAttributes(
        &attributes, <const void*>tensor_ptr
    )
    if status != cudaError.cudaSuccess:
        raise ValueError(
            f"Unable to determine CUDA device for array pointer: {status}"
        )
    return attributes.device


cdef int _dlpack_device_id_c(ary) except *:
    cdef uintptr_t tensor_ptr = <uintptr_t>ary.ai_["data"][0]
    if ary.from_cai:
        return _cuda_pointer_device_id(tensor_ptr)
    return 0


def _dlpack_device_id(ary):
    return _dlpack_device_id_c(ary)


cdef DLManagedTensor* dlpack_c(ary):
    # todo(dgd): add checking options/parameters
    cdef DLDeviceType dev_type
    cdef DLDevice dev
    cdef DLDataType dtype
    cdef DLTensor tensor
    cdef uintptr_t tensor_ptr = <uintptr_t>ary.ai_["data"][0]
    cdef DLManagedTensor* dlm = \
        <DLManagedTensor*>stdlib.malloc(sizeof(DLManagedTensor))

    if ary.from_cai:
        dev_type = DLDeviceType.kDLCUDA
    else:
        dev_type = DLDeviceType.kDLCPU

    dev.device_type = dev_type
    dev.device_id = _dlpack_device_id_c(ary)

    # todo (dgd): change to nice dict
    if ary.dtype == np.float32:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 32
    elif ary.dtype == np.float64:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 64
    elif ary.dtype == np.float16:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 16
    elif ary.dtype == np.int8:
        dtype.code = DLDataTypeCode.kDLInt
        dtype.bits = 8
    elif ary.dtype == np.int32:
        dtype.code = DLDataTypeCode.kDLInt
        dtype.bits = 32
    elif ary.dtype == np.int64:
        dtype.code = DLDataTypeCode.kDLInt
        dtype.bits = 64
    elif ary.dtype == np.uint8:
        dtype.code = DLDataTypeCode.kDLUInt
        dtype.bits = 8
    elif ary.dtype == np.uint32:
        dtype.code = DLDataTypeCode.kDLUInt
        dtype.bits = 32
    elif ary.dtype == np.uint64:
        dtype.code = DLDataTypeCode.kDLUInt
        dtype.bits = 64
    elif ary.dtype == np.bool_:
        dtype.code = DLDataTypeCode.kDLFloat
        dtype.bits = 8
    else:
        raise ValueError(f"Unsupported dtype {ary.dtype}")

    dtype.lanes = 1

    cdef size_t ndim = len(ary.shape)

    cdef int64_t* shape = <int64_t*>stdlib.malloc(ndim * sizeof(int64_t))

    for i in range(ndim):
        shape[i] = ary.shape[i]

    tensor.data = <void*> tensor_ptr
    tensor.device = dev
    tensor.dtype = dtype
    tensor.ndim = ndim
    tensor.shape = shape
    tensor.byte_offset = 0

    if ary.c_contiguous:
        tensor.strides = NULL
    elif ary.f_contiguous:
        tensor.strides = <int64_t*>stdlib.malloc(ndim * sizeof(int64_t))
        tensor.strides[0] = 1
        for i in range(1, ndim):
            tensor.strides[i] = tensor.strides[i-1] * tensor.shape[i-1]
    else:
        raise ValueError("Input data must be contiguous")

    dlm.dl_tensor = tensor
    dlm.manager_ctx = NULL
    dlm.deleter = deleter

    return dlm
