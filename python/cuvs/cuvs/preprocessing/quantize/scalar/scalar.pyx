#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array


cdef class QuantizerParams:
    """
    Parameters for scalar quantization

    Parameters
    ----------
    quantile: float
        specifies how many outliers at top & bottom will be ignored
        needs to be within range of (0, 1]
    """

    cdef cuvsScalarQuantizerParams * params

    def __cinit__(self):
        check_cuvs(cuvsScalarQuantizerParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsScalarQuantizerParamsDestroy(self.params))

    def __init__(self, *, quantile=None):
        if quantile is not None:
            self.params.quantile = quantile

    @property
    def quantile(self):
        return self.params.quantile


cdef class Quantizer:
    """
    Defines and stores scalar for quantisation upon training

    The quantization is performed by a linear mapping of an interval in the
    float data type to the full range of the quantized int type.
    """
    cdef cuvsScalarQuantizer * quantizer

    def __cinit__(self):
        check_cuvs(cuvsScalarQuantizerCreate(&self.quantizer))

    def __dealloc__(self):
        check_cuvs(cuvsScalarQuantizerDestroy(self.quantizer))

    @property
    def min(self):
        cdef double min_ = 0, max_ = 0
        check_cuvs(cuvsScalarQuantizerGetBounds(self.quantizer, &min_, &max_))
        return min_

    @property
    def max(self):
        cdef double min_ = 0, max_ = 0
        check_cuvs(cuvsScalarQuantizerGetBounds(self.quantizer, &min_, &max_))
        return max_

    def __repr__(self):
        return f"scalar.Quantizer(min={self.min}, max={self.max})"


@auto_sync_resources
def train(QuantizerParams params, dataset, resources=None):
    """
    Initializes a scalar quantizer to be used later for quantizing the dataset.

    Parameters
    ----------
    params : QuantizerParams object
    dataset : row major host or device dataset
    {resources_docstring}

    Returns
    -------
    quantizer: cuvs.preprocessing.quantize.scalar.Quantizer

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import scalar
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = scalar.QuantizerParams(quantile=0.99)
    >>> quantizer = scalar.train(params, dataset)
    >>> transformed = scalar.transform(quantizer, dataset)
    """
    dataset_ai = wrap_array(dataset)

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)

    _check_input_array(dataset_ai,
                       [np.dtype("float32"),
                        np.dtype("float64"),
                        np.dtype("float16")])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef Quantizer ret = Quantizer()

    check_cuvs(cuvsScalarQuantizerTrain(res,
                                        params.params,
                                        dataset_dlpack,
                                        ret.quantizer))

    return ret


@auto_sync_resources
@auto_convert_output
def transform(Quantizer quantizer, dataset, output=None, resources=None):
    """
    Applies quantization transform to given dataset

    Parameters
    ----------
    quantizer : trained Quantizer object
    dataset : row major host or device dataset to transform
    output : optional preallocated output memory, on host or device memory
    {resources_docstring}

    Returns
    -------
    output : transformed dataset quantized into a int8

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import scalar
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = scalar.QuantizerParams(quantile=0.99)
    >>> quantizer = scalar.train(params, dataset)
    >>> transformed = scalar.transform(quantizer, dataset)
    """

    dataset_ai = wrap_array(dataset)

    _check_input_array(dataset_ai,
                       [np.dtype("float32"),
                        np.dtype("float64"),
                        np.dtype("float16")])

    if output is None:
        on_device = hasattr(dataset, "__cuda_array_interface__")
        ndarray = device_ndarray if on_device else np
        output = ndarray.empty((dataset_ai.shape[0],
                                dataset_ai.shape[1]), dtype="int8")

    output_ai = wrap_array(output)
    _check_input_array(output_ai, [np.dtype("int8")])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cydlpack.DLManagedTensor* output_dlpack = \
        cydlpack.dlpack_c(output_ai)

    check_cuvs(cuvsScalarQuantizerTransform(res,
                                            quantizer.quantizer,
                                            dataset_dlpack,
                                            output_dlpack))

    return output


@auto_sync_resources
@auto_convert_output
def inverse_transform(Quantizer quantizer, dataset, output=None,
                      resources=None):
    """
    Perform inverse quantization step on previously quantized dataset

    Note that depending on the chosen data types train dataset the conversion
    is not lossless.

    Parameters
    ----------
    quantizer : trained Quantizer object
    dataset : row major host or device dataset to transform
    output : optional preallocated output memory, on host or device
    {resources_docstring}

    Returns
    -------
    output : transformed dataset with scalar quantization reversed
    """

    dataset_ai = wrap_array(dataset)

    _check_input_array(dataset_ai, [np.dtype("int8")])

    if output is None:
        on_device = hasattr(dataset, "__cuda_array_interface__")
        ndarray = device_ndarray if on_device else np
        output = ndarray.empty((dataset_ai.shape[0],
                                dataset_ai.shape[1]), dtype="float32")

    output_ai = wrap_array(output)
    _check_input_array(output_ai,
                       [np.dtype("float32"),
                        np.dtype("float64"),
                        np.dtype("float16")])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cydlpack.DLManagedTensor* output_dlpack = \
        cydlpack.dlpack_c(output_ai)

    check_cuvs(cuvsScalarQuantizerInverseTransform(res,
                                                   quantizer.quantizer,
                                                   dataset_dlpack,
                                                   output_dlpack))

    return output
