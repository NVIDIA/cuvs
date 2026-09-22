#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from collections import namedtuple

import cupy as cp
import numpy as np

from libcpp cimport bool

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources

SOLVER_TYPES = {
    "cov_eig_dq": cuvsPcaSolver.CUVS_PCA_COV_EIG_DQ,
    "cov_eig_jacobi": cuvsPcaSolver.CUVS_PCA_COV_EIG_JACOBI,
}

SOLVER_NAMES = {v: k for k, v in SOLVER_TYPES.items()}


cdef class Params:
    """
    Parameters for PCA decomposition.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep (default: 1).
    copy : bool
        If False, data passed to fit are overwritten and running
        fit(X) then transform(X) will not yield the expected results;
        use fit_transform(X) instead (default: True).
    whiten : bool
        When True the component vectors are multiplied by the square root
        of n_samples and divided by the singular values to ensure
        uncorrelated outputs with unit component-wise variances
        (default: False).
    algorithm : str
        Solver algorithm. One of ``"cov_eig_dq"`` (divide-and-conquer)
        or ``"cov_eig_jacobi"`` (Jacobi) (default: ``"cov_eig_dq"``).
    tol : float
        Tolerance for singular values, used by the Jacobi solver
        (default: 0.0).
    n_iterations : int
        Number of iterations for the Jacobi solver (default: 15).
    """

    cdef cuvsPcaParams* params

    def __cinit__(self):
        check_cuvs(cuvsPcaParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsPcaParamsDestroy(self.params))

    def __init__(self, *, n_components=None, copy=None, whiten=None,
                 algorithm=None, tol=None, n_iterations=None):
        if n_components is not None:
            self.params.n_components = n_components
        if copy is not None:
            self.params.copy = copy
        if whiten is not None:
            self.params.whiten = whiten
        if algorithm is not None:
            self.params.algorithm = <cuvsPcaSolver>SOLVER_TYPES[algorithm]
        if tol is not None:
            self.params.tol = tol
        if n_iterations is not None:
            self.params.n_iterations = n_iterations

    @property
    def n_components(self):
        return self.params.n_components

    @property
    def copy(self):
        return self.params.copy

    @property
    def whiten(self):
        return self.params.whiten

    @property
    def algorithm(self):
        return SOLVER_NAMES[self.params.algorithm]

    @property
    def tol(self):
        return self.params.tol

    @property
    def n_iterations(self):
        return self.params.n_iterations


FitOutput = namedtuple(
    "FitOutput",
    "components explained_var explained_var_ratio "
    "singular_vals mu noise_vars",
)

FitTransformOutput = namedtuple(
    "FitTransformOutput",
    "trans_input components explained_var explained_var_ratio "
    "singular_vals mu noise_vars",
)


def _ensure_device_contiguous(ary, dtype=np.dtype("float32")):
    """Ensure a device array is contiguous in either C- or F-order.

    Returns a tuple (arr, order) where ``order`` is ``"C"`` or ``"F"``.
    No copy is performed if the input is already contiguous in some order.
    """
    if hasattr(ary, "__cuda_array_interface__"):
        ary = cp.asarray(ary, dtype=dtype)
    else:
        ary = cp.asarray(np.asarray(ary, dtype=dtype))

    if ary.flags.c_contiguous:
        return ary, "C"
    if ary.flags.f_contiguous:
        return ary, "F"
    return cp.ascontiguousarray(ary), "C"


def _validate_pca_input(x_ai, expected_dtypes):
    """Verify ``x_ai`` has a supported dtype and is contiguous (C or F)."""
    if x_ai.dtype not in expected_dtypes:
        raise TypeError("dtype %s not supported" % x_ai.dtype)
    if not (x_ai.c_contiguous or x_ai.f_contiguous):
        raise ValueError("Input must be contiguous in C- or F-order")


@auto_sync_resources
def fit(Params params, X, resources=None):
    """
    Compute PCA (fit only).

    Computes the principal components, explained variances, singular
    values, and column means from the input data.

    The input layout (C-contiguous / row-major or F-contiguous / col-major)
    is preserved natively; no internal copy/transpose is performed. Output
    arrays use the same layout as the input.

    Parameters
    ----------
    params : Params
        PCA parameters. ``params.copy`` should be True if you intend
        to reuse *X* after this call.
    X : device array-like, shape (n_samples, n_features), float32
        Input data. Must be contiguous in either C- or F-order.
    {resources_docstring}

    Returns
    -------
    FitOutput
        Named tuple with fields: ``components``, ``explained_var``,
        ``explained_var_ratio``, ``singular_vals``, ``mu``,
        ``noise_vars``. ``components`` matches the layout of *X*.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8, copy=True)
    >>> result = pca.fit(params, X)
    >>> result.components.shape
    (8, 32)
    """
    n_components = params.n_components

    X_arr, order = _ensure_device_contiguous(X)
    x_ai = wrap_array(X_arr)
    _validate_pca_input(x_ai, [np.dtype("float32")])
    n_rows, n_cols = x_ai.shape

    components = cp.empty((n_components, n_cols), dtype="float32", order=order)
    explained_var = cp.empty((n_components,), dtype="float32")
    explained_var_ratio = cp.empty((n_components,), dtype="float32")
    singular_vals = cp.empty((n_components,), dtype="float32")
    mu = cp.empty((n_cols,), dtype="float32")
    noise_vars = cp.empty((1,), dtype="float32")

    cdef cydlpack.DLManagedTensor* x_dlpack = \
        cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* comp_dlpack = \
        cydlpack.dlpack_c(wrap_array(components))
    cdef cydlpack.DLManagedTensor* ev_dlpack = \
        cydlpack.dlpack_c(wrap_array(explained_var))
    cdef cydlpack.DLManagedTensor* evr_dlpack = \
        cydlpack.dlpack_c(wrap_array(explained_var_ratio))
    cdef cydlpack.DLManagedTensor* sv_dlpack = \
        cydlpack.dlpack_c(wrap_array(singular_vals))
    cdef cydlpack.DLManagedTensor* mu_dlpack = \
        cydlpack.dlpack_c(wrap_array(mu))
    cdef cydlpack.DLManagedTensor* nv_dlpack = \
        cydlpack.dlpack_c(wrap_array(noise_vars))

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsPcaFit(res, params.params, x_dlpack,
                           comp_dlpack, ev_dlpack, evr_dlpack,
                           sv_dlpack, mu_dlpack, nv_dlpack, False))

    return FitOutput(components, explained_var, explained_var_ratio,
                     singular_vals, mu, noise_vars)


@auto_sync_resources
def fit_transform(Params params, X, resources=None):
    """
    Compute PCA and transform the input data in a single operation.

    The input layout (C- or F-contiguous) is preserved natively; output
    arrays use the same layout.

    Parameters
    ----------
    params : Params
        PCA parameters.
    X : device array-like, shape (n_samples, n_features), float32
        Input data. Must be contiguous in either C- or F-order.
    {resources_docstring}

    Returns
    -------
    FitTransformOutput
        Named tuple with fields: ``trans_input``, ``components``,
        ``explained_var``, ``explained_var_ratio``, ``singular_vals``,
        ``mu``, ``noise_vars``. ``trans_input`` and ``components`` match
        the layout of *X*.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8)
    >>> result = pca.fit_transform(params, X)
    >>> result.trans_input.shape
    (500, 8)
    """
    n_components = params.n_components

    X_arr, order = _ensure_device_contiguous(X)
    x_ai = wrap_array(X_arr)
    _validate_pca_input(x_ai, [np.dtype("float32")])
    n_rows, n_cols = x_ai.shape

    trans_input = cp.empty((n_rows, n_components),
                           dtype="float32", order=order)
    components = cp.empty((n_components, n_cols), dtype="float32", order=order)
    explained_var = cp.empty((n_components,), dtype="float32")
    explained_var_ratio = cp.empty((n_components,), dtype="float32")
    singular_vals = cp.empty((n_components,), dtype="float32")
    mu = cp.empty((n_cols,), dtype="float32")
    noise_vars = cp.empty((1,), dtype="float32")

    cdef cydlpack.DLManagedTensor* x_dlpack = \
        cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* trans_dlpack = \
        cydlpack.dlpack_c(wrap_array(trans_input))
    cdef cydlpack.DLManagedTensor* comp_dlpack = \
        cydlpack.dlpack_c(wrap_array(components))
    cdef cydlpack.DLManagedTensor* ev_dlpack = \
        cydlpack.dlpack_c(wrap_array(explained_var))
    cdef cydlpack.DLManagedTensor* evr_dlpack = \
        cydlpack.dlpack_c(wrap_array(explained_var_ratio))
    cdef cydlpack.DLManagedTensor* sv_dlpack = \
        cydlpack.dlpack_c(wrap_array(singular_vals))
    cdef cydlpack.DLManagedTensor* mu_dlpack = \
        cydlpack.dlpack_c(wrap_array(mu))
    cdef cydlpack.DLManagedTensor* nv_dlpack = \
        cydlpack.dlpack_c(wrap_array(noise_vars))

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsPcaFitTransform(res, params.params, x_dlpack,
                                    trans_dlpack, comp_dlpack,
                                    ev_dlpack, evr_dlpack, sv_dlpack,
                                    mu_dlpack, nv_dlpack, False))

    return FitTransformOutput(trans_input, components, explained_var,
                              explained_var_ratio, singular_vals, mu,
                              noise_vars)


def _match_layout(arr, order):
    """Return ``arr`` in the requested layout (``"C"`` or ``"F"``).

    Avoids a copy when the array is already in the target layout."""
    arr = cp.asarray(arr)
    if order == "C" and arr.flags.c_contiguous:
        return arr
    if order == "F" and arr.flags.f_contiguous:
        return arr
    return cp.asarray(arr, order=order)


@auto_sync_resources
@auto_convert_output
def transform(Params params, X, components, singular_vals, mu,
              trans_input=None, resources=None):
    """
    Transform data into the PCA eigenspace.

    Uses previously computed principal components from :func:`fit` or
    :func:`fit_transform`. The input layout (C- or F-contiguous) of *X*
    determines the layout used internally; ``components`` and
    ``trans_input`` are aligned to that layout.

    Parameters
    ----------
    params : Params
        PCA parameters (must match those used during fit).
    X : device array-like, shape (n_samples, n_features), float32
        Data to transform.
    components : device array-like, shape (n_components, n_features)
        Principal components from a prior fit.
    singular_vals : device array-like, shape (n_components,)
        Singular values from a prior fit.
    mu : device array-like, shape (n_features,)
        Column means from a prior fit.
    trans_input : optional device array, shape (n_samples, n_components)
        Pre-allocated output buffer (float32). Layout is matched to *X*.
    {resources_docstring}

    Returns
    -------
    trans_input : device array, shape (n_samples, n_components)

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8, copy=True)
    >>> result = pca.fit(params, X)
    >>> transformed = pca.transform(params, X, result.components,
    ...                             result.singular_vals, result.mu)
    """
    n_components = params.n_components

    X_arr, order = _ensure_device_contiguous(X)
    x_ai = wrap_array(X_arr)
    _validate_pca_input(x_ai, [np.dtype("float32")])
    n_rows = x_ai.shape[0]

    components_arr = _match_layout(components, order)
    singular_vals_arr = cp.asarray(singular_vals)
    mu_arr = cp.asarray(mu)

    if trans_input is None:
        trans_input = cp.empty(
            (n_rows, n_components), dtype="float32", order=order
        )
    else:
        trans_input = _match_layout(trans_input, order)

    cdef cydlpack.DLManagedTensor* x_dlpack = \
        cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* comp_dlpack = \
        cydlpack.dlpack_c(wrap_array(components_arr))
    cdef cydlpack.DLManagedTensor* sv_dlpack = \
        cydlpack.dlpack_c(wrap_array(singular_vals_arr))
    cdef cydlpack.DLManagedTensor* mu_dlpack = \
        cydlpack.dlpack_c(wrap_array(mu_arr))
    cdef cydlpack.DLManagedTensor* trans_dlpack = \
        cydlpack.dlpack_c(wrap_array(trans_input))

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsPcaTransform(res, params.params, x_dlpack,
                                 comp_dlpack, sv_dlpack, mu_dlpack,
                                 trans_dlpack))

    return trans_input


@auto_sync_resources
@auto_convert_output
def inverse_transform(Params params, trans_input, components,
                      singular_vals, mu, output=None, resources=None):
    """
    Transform data from the PCA eigenspace back to the original space.

    The layout (C- or F-contiguous) of ``trans_input`` is preserved;
    ``components`` and ``output`` are aligned to that layout.

    Parameters
    ----------
    params : Params
        PCA parameters (must match those used during fit).
    trans_input : device array-like, shape (n_samples, n_components)
        Transformed data from :func:`transform` or :func:`fit_transform`.
    components : device array-like, shape (n_components, n_features)
        Principal components from a prior fit.
    singular_vals : device array-like, shape (n_components,)
        Singular values from a prior fit.
    mu : device array-like, shape (n_features,)
        Column means from a prior fit.
    output : optional device array, shape (n_samples, n_features)
        Pre-allocated output buffer (float32). Layout is matched to
        ``trans_input``.
    {resources_docstring}

    Returns
    -------
    output : device array, shape (n_samples, n_features)
        Reconstructed data.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8)
    >>> result = pca.fit_transform(params, X)
    >>> reconstructed = pca.inverse_transform(
    ...     params, result.trans_input, result.components,
    ...     result.singular_vals, result.mu)
    """
    trans_arr, order = _ensure_device_contiguous(trans_input)
    trans_ai = wrap_array(trans_arr)
    _validate_pca_input(trans_ai, [np.dtype("float32")])
    n_rows = trans_ai.shape[0]

    components_arr = _match_layout(components, order)
    comp_ai = wrap_array(components_arr)
    n_cols = comp_ai.shape[1]

    singular_vals_arr = cp.asarray(singular_vals)
    mu_arr = cp.asarray(mu)

    if output is None:
        output = cp.empty((n_rows, n_cols), dtype="float32", order=order)
    else:
        output = _match_layout(output, order)

    cdef cydlpack.DLManagedTensor* trans_dlpack = \
        cydlpack.dlpack_c(trans_ai)
    cdef cydlpack.DLManagedTensor* comp_dlpack = \
        cydlpack.dlpack_c(comp_ai)
    cdef cydlpack.DLManagedTensor* sv_dlpack = \
        cydlpack.dlpack_c(wrap_array(singular_vals_arr))
    cdef cydlpack.DLManagedTensor* mu_dlpack = \
        cydlpack.dlpack_c(wrap_array(mu_arr))
    cdef cydlpack.DLManagedTensor* out_dlpack = \
        cydlpack.dlpack_c(wrap_array(output))

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsPcaInverseTransform(res, params.params, trans_dlpack,
                                        comp_dlpack, sv_dlpack,
                                        mu_dlpack, out_dlpack))

    return output
