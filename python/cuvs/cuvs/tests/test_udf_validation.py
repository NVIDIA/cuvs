# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuvs._lib.udf_validation import (
    UnsupportedUDFCapture,
    UnsupportedUDFSyntax,
    validate_capture,
    validate_ivf_flat_metadata,
    validate_signature,
    validate_udf_policy,
)


def valid_l2(x, y, acc, ctx):
    d = x - y
    return acc + d * d


def valid_ctx_read(x, y, acc, ctx):
    d = x - y
    return acc + ctx.weights[ctx.dim] * d * d


def invalid_call(x, y, acc, ctx):
    return acc + ctx.np.sqrt(x - y)


def invalid_ctx_write(x, y, acc, ctx):
    ctx.weights[ctx.dim] = 1.0
    return acc


def invalid_for_iter(x, y, acc, ctx):
    for item in ctx.weights:
        acc = acc + item
    return acc


def valid_range_loop(x, y, acc, ctx):
    for i in range(2):
        acc = acc + i * (x - y)
    return acc


def test_validate_signature_accepts_expected_positional_args():
    validate_signature(valid_l2, ["x", "y", "acc", "ctx"])


def test_validate_signature_rejects_wrong_names_and_defaults():
    def wrong(a, b, c, d):
        return a + b + c + d

    def default_arg(x, y, acc, ctx=None):
        return acc

    with pytest.raises(TypeError, match="expected f"):
        validate_signature(wrong, ["x", "y", "acc", "ctx"])

    with pytest.raises(TypeError, match="defaults"):
        validate_signature(default_arg, ["x", "y", "acc", "ctx"])


def test_validate_signature_rejects_varargs_and_keyword_only():
    def varargs(x, y, acc, ctx, *rest):
        return acc

    def keyword_only(x, y, acc, *, ctx):
        return acc

    with pytest.raises(TypeError, match="keyword-only"):
        validate_signature(varargs, ["x", "y", "acc", "ctx", "rest"])

    with pytest.raises(TypeError, match="keyword-only"):
        validate_signature(keyword_only, ["x", "y", "acc", "ctx"])


def test_validate_ivf_flat_metadata():
    validate_ivf_flat_metadata("min", 0.0, "sqeuclidean")

    with pytest.raises(ValueError, match="order"):
        validate_ivf_flat_metadata("max", 0.0, "sqeuclidean")

    with pytest.raises(TypeError, match="initial"):
        validate_ivf_flat_metadata("min", object(), "sqeuclidean")

    with pytest.raises(ValueError, match="coarse_metric"):
        validate_ivf_flat_metadata("min", 0.0, "cosine")


def test_validate_udf_policy_accepts_basic_subset():
    validate_udf_policy(valid_l2)
    validate_udf_policy(valid_ctx_read)
    validate_udf_policy(valid_range_loop)


def test_validate_udf_policy_rejects_arbitrary_call():
    with pytest.raises(UnsupportedUDFSyntax, match="ctx.np.sqrt"):
        validate_udf_policy(invalid_call)


def test_validate_udf_policy_rejects_capture_mutation():
    with pytest.raises(UnsupportedUDFSyntax, match="read-only"):
        validate_udf_policy(invalid_ctx_write)


def test_validate_udf_policy_rejects_non_range_for_loop():
    with pytest.raises(UnsupportedUDFSyntax, match="range"):
        validate_udf_policy(invalid_for_iter)


def test_validate_capture_accepts_cupy_cuda_array_interface():
    cp = pytest.importorskip("cupy")

    with cp.cuda.Device(0):
        owner = cp.arange(8, dtype=cp.float32)
        capture = validate_capture("weights", owner, expected_device=0)

    assert capture.name == "weights"
    assert capture.dtype == "float32"
    assert capture.shape == (8,)
    # CuPy omits strides from CUDA Array Interface for contiguous arrays.
    assert capture.strides is None
    assert capture.pointer == owner.data.ptr
    assert capture.owner is owner
    assert capture.readonly is True


def test_validate_capture_accepts_cupy_strided_view():
    cp = pytest.importorskip("cupy")

    with cp.cuda.Device(0):
        owner = cp.arange(16, dtype=cp.float32)[::2]
        capture = validate_capture("weights", owner, expected_device=0)

    assert capture.shape == owner.shape
    assert capture.strides == owner.strides
    assert capture.pointer == owner.data.ptr


def test_validate_capture_rejects_missing_cai():
    with pytest.raises(UnsupportedUDFCapture, match="CUDA Array Interface"):
        validate_capture("weights", object())


def test_validate_capture_rejects_unsupported_cupy_dtype():
    cp = pytest.importorskip("cupy")

    with cp.cuda.Device(0):
        owner = cp.arange(8, dtype=cp.float64)

    with pytest.raises(UnsupportedUDFCapture, match="unsupported capture dtype"):
        validate_capture("weights", owner, expected_device=0)


def test_validate_capture_rejects_wrong_expected_device():
    cp = pytest.importorskip("cupy")

    with cp.cuda.Device(0):
        owner = cp.arange(8, dtype=cp.float32)

    with pytest.raises(UnsupportedUDFCapture, match="device"):
        validate_capture("weights", owner, expected_device=1)
