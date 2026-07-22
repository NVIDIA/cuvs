# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuvs._lib.device_udf import UDFTarget
from cuvs._lib.udf_backends.numba_cuda_mlir import NumbaCudaMLIRBackend


def weighted_l2_update(x, y, acc, weights, dim):
    d = x - y
    return acc + weights[dim] * d * d


def l2_update(x, y, acc):
    d = x - y
    return acc + d * d


@pytest.mark.parametrize("fn,symbol,arg_builder", [
    (
        l2_update,
        "cuvs_l2_update_f32_backend_test",
        lambda backend: (
            backend.float32(),
            backend.float32(),
            backend.float32(),
        ),
    ),
    (
        weighted_l2_update,
        "cuvs_weighted_l2_update_f32_backend_test",
        lambda backend: (
            backend.float32(),
            backend.float32(),
            backend.float32(),
            backend.float32_pointer(),
            backend.int64(),
        ),
    ),
])
def test_numba_cuda_mlir_backend_compiles_ltoir_artifact(fn, symbol, arg_builder):
    pytest.importorskip("numba_cuda_mlir")

    backend = NumbaCudaMLIRBackend()
    target = UDFTarget(
        sm="120",
        cuda_version="13.2",
        nvrtc_version=None,
        nvjitlink_version="13.2",
        numba_cuda_mlir_version="0.3.0",
        compile_options=("-lto",),
    )

    result = backend.compile(
        fn,
        abi="rapids.cuvs.ivf_flat.metric.v1",
        symbol_name=symbol,
        arg_types=arg_builder(backend),
        return_type=backend.float32(),
        target=target,
        lowering_version="explicit-lowered-v0",
        algorithm_options={"adapter": "ivf_flat", "dtype": "float32"},
    )

    artifact = result.artifact
    assert artifact.payload_kind == "ltoir"
    assert artifact.symbol_name == symbol
    assert artifact.abi == "rapids.cuvs.ivf_flat.metric.v1"
    assert artifact.payload_bytes().startswith(b"\xedCN")
    assert len(artifact.payload_bytes()) > 1024
    assert artifact.cache_key.startswith("rapids.cuvs.device_udf.cache_key.v1:")
    assert result.return_type == backend.float32()



class FakeType:
    def __init__(self, name):
        self.name = name

    def __call__(self, *arg_types):
        return (self, arg_types)

    def __repr__(self):
        return self.name


class FakeTypes:
    float32 = FakeType("float32")
    int64 = FakeType("int64")


class FakeCuda:
    def compile(self, *args, **kwargs):
        return b"fake-ltoir", FakeTypes.int64


def test_numba_cuda_mlir_backend_rejects_return_type_mismatch():
    backend = NumbaCudaMLIRBackend(cuda_module=FakeCuda(), types_module=FakeTypes())
    target = UDFTarget(
        sm="120",
        cuda_version="13.2",
        nvrtc_version=None,
        nvjitlink_version="13.2",
        numba_cuda_mlir_version="0.3.0",
        compile_options=("-lto",),
    )

    with pytest.raises(TypeError, match="return type"):
        backend.compile(
            l2_update,
            abi="rapids.cuvs.ivf_flat.metric.v1",
            symbol_name="cuvs_l2_update_return_mismatch",
            arg_types=(FakeTypes.float32, FakeTypes.float32, FakeTypes.float32),
            return_type=FakeTypes.float32,
            target=target,
            lowering_version="explicit-lowered-v0",
        )
