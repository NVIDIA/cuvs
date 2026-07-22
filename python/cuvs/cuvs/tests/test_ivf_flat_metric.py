# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuvs._lib.udf_validation import UnsupportedUDFCapture, validate_capture
from cuvs.neighbors import ivf_flat
from cuvs.neighbors.ivf_flat._udf import _function_source, _lower_metric


class FakeCudaArray:
    def __init__(
        self, *, typestr="<f4", shape=(4,), strides=None, pointer=1234
    ):
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": typestr,
            "data": (pointer, False),
            "strides": strides,
        }


def test_metric_decorator_rejects_non_cuda_capture_before_compile():
    with pytest.raises(UnsupportedUDFCapture, match="CUDA Array Interface"):

        @ivf_flat.metric(captures={"weights": object()})
        def weighted_l2(x, y, acc, ctx):
            d = x - y
            return acc + d * d


def test_metric_decorator_rejects_ctx_access_without_capture_before_compile():
    with pytest.raises(NotImplementedError, match="ctx access"):

        @ivf_flat.metric()
        def weighted_l2(x, y, acc, ctx):
            d = x - y
            return acc + ctx.weights[ctx.dim] * d * d


def test_metric_decorator_rejects_unknown_ctx_capture_before_compile():
    with pytest.raises(UnsupportedUDFCapture, match="ctx.bias"):

        @ivf_flat.metric(captures={"weights": FakeCudaArray()})
        def weighted_l2(x, y, acc, ctx):
            d = x - y
            return acc + ctx.bias[ctx.dim] * d * d


def test_metric_decorator_rejects_strided_capture_before_compile():
    with pytest.raises(UnsupportedUDFCapture, match="contiguous"):

        @ivf_flat.metric(captures={"weights": FakeCudaArray(strides=(8,))})
        def weighted_l2(x, y, acc, ctx):
            d = x - y
            return acc + ctx.weights[ctx.dim] * d * d


def test_metric_lowering_desugars_capture_and_dim():
    def weighted_l2(x, y, acc, ctx):
        d = x - y
        return acc + ctx.weights[ctx.dim] * d * d

    capture = validate_capture("weights", FakeCudaArray())
    lowered_fn, lowered_source = _lower_metric(
        weighted_l2,
        "cuvs_test_weighted_l2",
        _function_source(weighted_l2),
        (capture,),
    )

    assert "ctx" not in lowered_source
    assert "weights[dim]" in lowered_source
    assert lowered_fn(3.0, 1.0, 2.0, [0.5], 0) == 4.0


def test_metric_decorator_validates_signature_before_compile():
    with pytest.raises(TypeError, match="expected f"):

        @ivf_flat.metric()
        def l2_update(x, y, acc):
            d = x - y
            return acc + d * d
