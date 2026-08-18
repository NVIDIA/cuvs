#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

from cuvs.common cimport cydlpack
from cuvs.common.dataset cimport (
    Dataset,
    cuvsDatasetMakeStandardView,
    cuvsDatasetMakeVpq,
)
from cuvs.neighbors.cagra.cagra cimport (
    cuvsCagraCompressionParams,
    cuvsCagraCompressionParamsCreate,
    cuvsCagraCompressionParamsDestroy,
)

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.device_tensor_view import DeviceTensorView
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array

PQ_KMEANS_TYPES = {
    "kmeans" : cuvsKMeansType.CUVS_KMEANS_TYPE_KMEANS,
    "kmeans_balanced" : cuvsKMeansType.CUVS_KMEANS_TYPE_KMEANS_BALANCED}

PQ_KMEANS_NAMES = {v: k for k, v in PQ_KMEANS_TYPES.items()}


cdef class VpqParams:
    """Parameters for creating a CAGRA-Q VPQ dataset."""

    cdef cuvsCagraCompressionParams* params

    def __cinit__(self):
        self.params = NULL
        check_cuvs(cuvsCagraCompressionParamsCreate(&self.params))

    def __dealloc__(self):
        if self.params != NULL:
            cuvsCagraCompressionParamsDestroy(self.params)

    def __init__(self, *, pq_bits=8, pq_dim=0, vq_n_centers=0,
                 kmeans_n_iters=25, vq_kmeans_trainset_fraction=0.0,
                 pq_kmeans_trainset_fraction=0.0):
        self.params.pq_bits = pq_bits
        self.params.pq_dim = pq_dim
        self.params.vq_n_centers = vq_n_centers
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.vq_kmeans_trainset_fraction = \
            vq_kmeans_trainset_fraction
        self.params.pq_kmeans_trainset_fraction = \
            pq_kmeans_trainset_fraction

cdef class QuantizerParams:
    """
    Parameters for product quantization

    Parameters
    ----------
    pq_bits: int
        specifies the bit length of the vector element after compression by PQ
        possible values: within [4, 16]
    pq_dim: int
        specifies the dimensionality of the vector after compression by PQ
    use_subspaces: bool
        specifies whether to use subspaces for product quantization (PQ).
        When true, one PQ codebook is used for each subspace. Otherwise, a
        single PQ codebook is used.
    use_vq: bool
        specifies whether to use Vector Quantization (KMeans) before product
        quantization (PQ).
    vq_n_centers: int
        specifies the number of centers for the vector quantizer.
        When zero, an optimal value is selected using a heuristic.
        When one, only product quantization is used.
    kmeans_n_iters: int
        specifies the number of iterations searching for kmeans centers
    pq_kmeans_type: str
        specifies the type of kmeans algorithm to use for PQ training
        possible values: "kmeans", "kmeans_balanced"
    max_train_points_per_pq_code: int
        specifies the max number of data points to use per PQ code during PQ
        codebook training. Using more data points per PQ code may increase the
        quality of PQ codebook but may also increase the build time.
    max_train_points_per_vq_cluster: int
        specifies the max number of data points to use per VQ cluster.
    """

    cdef cuvsProductQuantizerParams * params

    def __cinit__(self):
        check_cuvs(cuvsProductQuantizerParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsProductQuantizerParamsDestroy(self.params))

    def __init__(self, *, pq_bits=8, pq_dim=0, use_subspaces=True,
                 use_vq=False, vq_n_centers=0, kmeans_n_iters=25,
                 pq_kmeans_type="kmeans_balanced",
                 max_train_points_per_pq_code=256,
                 max_train_points_per_vq_cluster=1024):
        if pq_bits < 4 or pq_bits > 16:
            raise ValueError("pq_bits must be within [4, 16]")
        self.params.pq_bits = pq_bits
        self.params.pq_dim = pq_dim
        self.params.use_subspaces = use_subspaces
        self.params.use_vq = use_vq
        self.params.vq_n_centers = vq_n_centers
        self.params.kmeans_n_iters = kmeans_n_iters
        pq_kmeans_type_c = PQ_KMEANS_TYPES[pq_kmeans_type]
        self.params.pq_kmeans_type = <cuvsKMeansType>pq_kmeans_type_c
        self.params.max_train_points_per_pq_code = max_train_points_per_pq_code
        self.params.max_train_points_per_vq_cluster = max_train_points_per_vq_cluster

    @property
    def pq_bits(self):
        return self.params.pq_bits

    @property
    def pq_dim(self):
        return self.params.pq_dim

    @property
    def vq_n_centers(self):
        return self.params.vq_n_centers

    @property
    def kmeans_n_iters(self):
        return self.params.kmeans_n_iters

    @property
    def pq_kmeans_type(self):
        return self.params.pq_kmeans_type

    @property
    def max_train_points_per_pq_code(self):
        return self.params.max_train_points_per_pq_code

    @property
    def max_train_points_per_vq_cluster(self):
        return self.params.max_train_points_per_vq_cluster

    @property
    def use_vq(self):
        return self.params.use_vq

    @property
    def use_subspaces(self):
        return self.params.use_subspaces


cdef class Quantizer:
    """
    Defines and stores Product Quantizer upon training

    The quantization is performed by a linear mapping of an interval in the
    float data type to the full range of the quantized int type.
    """
    cdef cuvsProductQuantizer * quantizer

    def __cinit__(self):
        check_cuvs(cuvsProductQuantizerCreate(&self.quantizer))

    def __dealloc__(self):
        check_cuvs(cuvsProductQuantizerDestroy(self.quantizer))

    @property
    def pq_bits(self):
        cdef uint32_t pq_bits
        check_cuvs(cuvsProductQuantizerGetPqBits(self.quantizer, &pq_bits))
        return pq_bits

    @property
    def pq_dim(self):
        cdef uint32_t pq_dim
        check_cuvs(cuvsProductQuantizerGetPqDim(self.quantizer, &pq_dim))
        return pq_dim

    @property
    def pq_codebook(self):
        """
        Returns the PQ codebook
        """
        output = DeviceTensorView()
        cdef cydlpack.DLManagedTensor* pq_codebook_dlpack = \
            <cydlpack.DLManagedTensor*><size_t>output.get_handle()
        check_cuvs(cuvsProductQuantizerGetPqCodebook(self.quantizer,
                                                     pq_codebook_dlpack))
        output.parent = self
        return output

    @property
    def vq_codebook(self):
        """
        Returns the VQ codebook
        """
        output = DeviceTensorView()
        cdef cydlpack.DLManagedTensor* vq_codebook_dlpack = \
            <cydlpack.DLManagedTensor*><size_t>output.get_handle()
        check_cuvs(cuvsProductQuantizerGetVqCodebook(self.quantizer,
                                                     vq_codebook_dlpack))
        output.parent = self
        return output

    @property
    def encoded_dim(self):
        """
        Returns the encoded dimension of the quantized dataset
        """
        cdef uint32_t encoded_dim
        check_cuvs(cuvsProductQuantizerGetEncodedDim(self.quantizer,
                                                     &encoded_dim))
        return encoded_dim

    @property
    def use_vq(self):
        cdef bool use_vq
        check_cuvs(cuvsProductQuantizerGetUseVq(self.quantizer, &use_vq))
        return use_vq

    def __repr__(self):
        return f"pq.Quantizer(pq_bits={self.pq_bits}, " \
               f"pq_dim={self.pq_dim})"


@auto_sync_resources
def build(QuantizerParams params, dataset, resources=None):
    """
    Builds a Product Quantizer to be used later for quantizing
    the dataset.

    Parameters
    ----------
    params : QuantizerParams object
    dataset : row major dataset on host or device memory. FP32
    {resources_docstring}

    Returns
    -------
    quantizer: cuvs.preprocessing.quantize.pq.Quantizer

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import pq
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16)
    >>> quantizer = pq.build(params, dataset)
    >>> transformed, _ = pq.transform(quantizer, dataset)
    """
    dataset_ai = wrap_array(dataset)

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)

    _check_input_array(dataset_ai,
                       [np.dtype("float32")])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef Quantizer ret = Quantizer()

    check_cuvs(cuvsProductQuantizerBuild(res,
                                         params.params,
                                         dataset_dlpack,
                                         ret.quantizer))

    return ret


@auto_sync_resources
@auto_convert_output
def transform(Quantizer quantizer, dataset, codes_output=None, vq_labels=None, resources=None):
    """
    Applies Product Quantization transform to given dataset

    Parameters
    ----------
    quantizer : trained Quantizer object
    dataset : row major dataset on host or device memory. FP32
    codes_output : optional preallocated output memory, on device memory
    vq_labels : optional preallocated output memory for VQ labels, on device memory
    {resources_docstring}

    Returns
    -------
    codes_output : transformed dataset quantized into a uint8
    vq_labels : VQ labels when VQ is used, None otherwise

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import pq
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16)
    >>> quantizer = pq.build(params, dataset)
    >>> transformed, _ = pq.transform(quantizer, dataset)
    """

    dataset_ai = wrap_array(dataset)

    _check_input_array(dataset_ai,
                       [np.dtype("float32")])

    if codes_output is None:
        encoded_cols = quantizer.encoded_dim
        codes_output = device_ndarray.empty((dataset_ai.shape[0],
                                       encoded_cols), dtype="uint8")

    codes_output_ai = wrap_array(codes_output)
    _check_input_array(codes_output_ai, [np.dtype("uint8")])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cydlpack.DLManagedTensor* codes_output_dlpack = \
        cydlpack.dlpack_c(codes_output_ai)
    cdef cydlpack.DLManagedTensor* vq_labels_dlpack = NULL
    if quantizer.use_vq:
        if vq_labels is None:
            vq_labels = device_ndarray.empty((dataset_ai.shape[0],), dtype="uint32")
        _check_input_array(vq_labels, [np.dtype("uint32")])
        vq_labels_dlpack = cydlpack.dlpack_c(wrap_array(vq_labels))

    check_cuvs(cuvsProductQuantizerTransform(res,
                                             quantizer.quantizer,
                                             dataset_dlpack,
                                             codes_output_dlpack,
                                             vq_labels_dlpack))

    return codes_output, vq_labels


@auto_sync_resources
@auto_convert_output
def inverse_transform(Quantizer quantizer, codes, output=None, vq_labels=None, resources=None):
    """
    Applies Product Quantization inverse transform to given codes

    Parameters
    ----------
    quantizer : trained Quantizer object
    codes : row major device codes to inverse transform. uint8
    output : optional preallocated output memory, on device memory
    vq_labels : optional VQ labels when VQ is used, on device memory
    {resources_docstring}

    Returns
    -------
    output : Original dataset reconstructed from quantized codes

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing.quantize import pq
    >>> n_samples = 5000
    >>> n_features = 64
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> params = pq.QuantizerParams(pq_bits=8, pq_dim=16, use_vq=True)
    >>> quantizer = pq.build(params, dataset)
    >>> transformed, vq_labels = pq.transform(quantizer, dataset)
    >>> reconstructed = pq.inverse_transform(quantizer, transformed, vq_labels=vq_labels)
    """

    if quantizer.use_vq and vq_labels is None:
        raise ValueError("vq_labels must be provided when VQ is used")
    codes_ai = wrap_array(codes)

    _check_input_array(codes_ai, [np.dtype("uint8")])
    pq_cdbk = quantizer.pq_codebook

    if output is None:
        on_device = hasattr(codes, "__cuda_array_interface__")
        ndarray = device_ndarray if on_device else np
        original_cols = quantizer.pq_dim * pq_cdbk.shape[1]
        output = ndarray.empty((codes_ai.shape[0],
                                original_cols), dtype=pq_cdbk.dtype)

    cdef cydlpack.DLManagedTensor* vq_labels_dlpack = NULL

    if vq_labels is not None:
        _check_input_array(vq_labels, [np.dtype("uint32")])
        vq_labels_dlpack = cydlpack.dlpack_c(wrap_array(vq_labels))
    output_ai = wrap_array(output)
    _check_input_array(output_ai, [pq_cdbk.dtype])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* codes_dlpack = \
        cydlpack.dlpack_c(codes_ai)
    cdef cydlpack.DLManagedTensor* output_dlpack = \
        cydlpack.dlpack_c(output_ai)

    check_cuvs(cuvsProductQuantizerInverseTransform(res,
                                                    quantizer.quantizer,
                                                    codes_dlpack,
                                                    output_dlpack,
                                                    vq_labels_dlpack))

    return output


@auto_sync_resources
def make_vpq_dataset(VpqParams params, dataset, resources=None):
    """Create an owning device VPQ dataset for iterative CAGRA-Q."""
    cdef Dataset dense
    cdef Dataset vpq = Dataset()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = NULL

    if isinstance(dataset, Dataset):
        dense = dataset
    else:
        dataset_ai = wrap_array(dataset)
        _check_input_array(
            dataset_ai,
            [np.dtype("float32"), np.dtype("float16"),
             np.dtype("int8"), np.dtype("uint8")])
        dataset_dlpack = cydlpack.dlpack_c(dataset_ai)
        dense = Dataset()
        check_cuvs(cuvsDatasetMakeStandardView(
            res, dataset_dlpack, &dense.dataset))

    check_cuvs(cuvsDatasetMakeVpq(
        res, params.params, dense.dataset, &vpq.dataset))
    return vpq
