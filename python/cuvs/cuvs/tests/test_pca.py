# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest

from cuvs.preprocessing import pca


def _as_order(arr, order):
    if order == "C":
        return cp.ascontiguousarray(arr)
    return cp.asfortranarray(arr)


@pytest.mark.parametrize("n_rows", [256, 512])
@pytest.mark.parametrize("n_cols", [32, 64])
@pytest.mark.parametrize("order", ["C", "F"])
def test_fit_transform_inverse_transform(n_rows, n_cols, order):
    """
    fit_transform with all components then inverse_transform
    should reconstruct the original data near-losslessly, regardless of
    whether the input is C- or F-contiguous.
    """
    n_components = n_cols
    X = _as_order(
        cp.random.random_sample((n_rows, n_cols), dtype=cp.float32), order
    )

    params = pca.Params(n_components=n_components)
    result = pca.fit_transform(params, X)

    assert result.trans_input.shape == (n_rows, n_components)
    assert result.components.shape == (n_components, n_cols)

    expected_c = order == "C"
    assert result.trans_input.flags.c_contiguous == expected_c
    assert result.trans_input.flags.f_contiguous == (not expected_c)
    assert result.components.flags.c_contiguous == expected_c
    assert result.components.flags.f_contiguous == (not expected_c)

    reconstructed = pca.inverse_transform(
        params,
        result.trans_input,
        result.components,
        result.singular_vals,
        result.mu,
    )

    max_err = float(cp.max(cp.abs(X - reconstructed)))
    assert max_err < 1e-3, (
        f"Reconstruction error {max_err} too large for lossless case "
        f"(order={order})"
    )


@pytest.mark.parametrize("n_rows", [256, 512])
@pytest.mark.parametrize("n_cols", [32, 64])
@pytest.mark.parametrize("order", ["C", "F"])
def test_fit_then_transform(n_rows, n_cols, order):
    """
    fit() then transform() separately should give the same result
    as fit_transform() when copy=True, regardless of input layout.
    """
    n_components = n_cols
    X = _as_order(
        cp.random.random_sample((n_rows, n_cols), dtype=cp.float32), order
    )

    params = pca.Params(n_components=n_components, copy=True)
    fit_result = pca.fit(params, X)

    assert fit_result.components.shape == (n_components, n_cols)
    assert fit_result.singular_vals.shape == (n_components,)
    assert fit_result.mu.shape == (n_cols,)

    transformed = pca.transform(
        params,
        X,
        fit_result.components,
        fit_result.singular_vals,
        fit_result.mu,
    )
    assert transformed.shape == (n_rows, n_components)

    reconstructed = pca.inverse_transform(
        params,
        transformed,
        fit_result.components,
        fit_result.singular_vals,
        fit_result.mu,
    )

    max_err = float(cp.max(cp.abs(X - reconstructed)))
    assert max_err < 1e-3, (
        f"Reconstruction error {max_err} too large for lossless case "
        f"(order={order})"
    )


@pytest.mark.parametrize("n_rows", [512])
@pytest.mark.parametrize("n_cols", [64])
@pytest.mark.parametrize("n_components", [8, 16])
@pytest.mark.parametrize("order", ["C", "F"])
def test_dim_reduction(n_rows, n_cols, n_components, order):
    """With fewer components, reconstruction should have bounded error."""
    X = _as_order(
        cp.random.random_sample((n_rows, n_cols), dtype=cp.float32), order
    )

    params = pca.Params(n_components=n_components)
    result = pca.fit_transform(params, X)

    assert result.trans_input.shape == (n_rows, n_components)

    reconstructed = pca.inverse_transform(
        params,
        result.trans_input,
        result.components,
        result.singular_vals,
        result.mu,
    )

    max_err = float(cp.max(cp.abs(X - reconstructed)))
    assert max_err > 1e-5, (
        "Reconstruction error should be non-zero with fewer components"
    )
    assert max_err < 2.0, f"Reconstruction error {max_err} should be bounded"


def test_row_major_no_copy():
    """
    When the user passes a row-major (C-contiguous) array, the data pointer
    must be preserved -- i.e. the implementation must NOT silently reorder
    the input into Fortran layout.
    """
    n_rows, n_cols = 128, 16
    X = cp.random.random_sample((n_rows, n_cols), dtype=cp.float32)
    assert X.flags.c_contiguous

    original_ptr = X.data.ptr

    params = pca.Params(n_components=8)
    pca.fit(params, X)

    # The user's array was contiguous and float32 -- the C ABI should accept
    # it as-is, not reallocate or reorder it.
    assert X.data.ptr == original_ptr
    assert X.flags.c_contiguous


def test_layouts_agree_numerically():
    """
    Running PCA on the same data in C-order and F-order should yield the
    same explained variances (within tolerance), and the same reconstruction.
    Components individually may differ in sign convention but reconstruction
    is invariant to sign flip of components.
    """
    n_rows, n_cols, n_components = 256, 32, 16
    rng = cp.random.default_rng(42)
    X = rng.standard_normal((n_rows, n_cols), dtype=cp.float32)

    X_c = cp.ascontiguousarray(X)
    X_f = cp.asfortranarray(X)

    params = pca.Params(n_components=n_components)
    res_c = pca.fit_transform(params, X_c)
    res_f = pca.fit_transform(params, X_f)

    cp.testing.assert_allclose(
        res_c.explained_var, res_f.explained_var, rtol=1e-4, atol=1e-4
    )
    cp.testing.assert_allclose(
        res_c.singular_vals, res_f.singular_vals, rtol=1e-4, atol=1e-4
    )
    cp.testing.assert_allclose(res_c.mu, res_f.mu, rtol=1e-5, atol=1e-5)

    recon_c = pca.inverse_transform(
        params,
        res_c.trans_input,
        res_c.components,
        res_c.singular_vals,
        res_c.mu,
    )
    recon_f = pca.inverse_transform(
        params,
        res_f.trans_input,
        res_f.components,
        res_f.singular_vals,
        res_f.mu,
    )
    cp.testing.assert_allclose(recon_c, recon_f, rtol=1e-3, atol=1e-3)


def test_explained_variance():
    """
    When all components are kept, explained_var_ratio should sum
    to approximately 1.0.
    """
    n_rows, n_cols = 512, 32
    X = cp.random.random_sample((n_rows, n_cols), dtype=cp.float32)

    params = pca.Params(n_components=n_cols)
    result = pca.fit(params, X)

    ratio_sum = float(cp.sum(result.explained_var_ratio))
    assert abs(ratio_sum - 1.0) < 0.05, (
        f"Explained variance ratio sum {ratio_sum} not close to 1.0"
    )

    assert cp.all(result.explained_var >= 0), (
        "Explained variances should be non-negative"
    )
    assert cp.all(result.singular_vals >= 0), (
        "Singular values should be non-negative"
    )


def test_params_defaults():
    """Test that Params has sensible defaults."""
    p = pca.Params()
    assert p.n_components == 1
    assert p.copy is True
    assert p.whiten is False
    assert p.algorithm == "cov_eig_dq"
    assert p.tol == 0.0
    assert p.n_iterations == 15


def test_params_custom():
    """Test that custom Params values propagate correctly."""
    p = pca.Params(
        n_components=10,
        copy=False,
        whiten=True,
        algorithm="cov_eig_jacobi",
        tol=1e-4,
        n_iterations=50,
    )
    assert p.n_components == 10
    assert p.copy is False
    assert p.whiten is True
    assert p.algorithm == "cov_eig_jacobi"
    assert abs(p.tol - 1e-4) < 1e-7
    assert p.n_iterations == 50
