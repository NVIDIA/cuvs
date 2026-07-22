# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuvs._lib.device_udf import (
    UDFCapture,
    UDFTarget,
    build_cache_key,
    make_udf_artifact,
    source_hash,
)


def _target(**kwargs):
    values = {
        "sm": "120",
        "cuda_version": "13.2",
        "nvrtc_version": "13.2",
        "nvjitlink_version": "13.2",
        "numba_cuda_mlir_version": "0.3.0",
        "compile_options": ("-lto",),
    }
    values.update(kwargs)
    return UDFTarget(**values)


def _capture(**kwargs):
    values = {
        "name": "weights",
        "dtype": "float32",
        "shape": (128,),
        "strides": None,
        "device_id": 0,
        "readonly": True,
        "pointer": 1234,
        "owner": object(),
    }
    values.update(kwargs)
    return UDFCapture(**values)


def _key(**kwargs):
    values = {
        "abi": "rapids.cuvs.ivf_flat.metric.v1",
        "payload_kind": "ltoir",
        "payload": b"fake-ltoir",
        "target": _target(),
        "captures": (_capture(),),
        "source_hash": source_hash("def f(x, y, acc): return acc + (x - y) * (x - y)"),
        "lowering_version": "ctx-lowering-v1",
        "algorithm_options": {"adapter": "ivf_flat", "dtype": "float32"},
    }
    values.update(kwargs)
    return build_cache_key(**values)


def _artifact(**kwargs):
    values = {
        "abi": "rapids.cuvs.ivf_flat.metric.v1",
        "payload_kind": "ltoir",
        "payload": b"fake-ltoir",
        "symbol_name": "cuvs_l2_update_f32",
        "captures": (),
        "target": _target(),
        "source_hash": source_hash("def f(x, y, acc): return acc"),
        "lowering_version": "ctx-lowering-v1",
        "algorithm_options": {"adapter": "ivf_flat", "dtype": "float32"},
    }
    values.update(kwargs)
    return make_udf_artifact(**values)


def test_cache_key_is_stable_for_same_metadata():
    assert _key() == _key()


def test_cache_key_excludes_capture_pointer_and_owner():
    left = _capture(pointer=1234, owner=object())
    right = _capture(pointer=987654, owner=object())

    assert _key(captures=(left,)) == _key(captures=(right,))


def test_cache_key_includes_capture_metadata():
    assert _key(captures=(_capture(dtype="float32"),)) != _key(
        captures=(_capture(dtype="int32"),)
    )
    assert _key(captures=(_capture(shape=(128,)),)) != _key(
        captures=(_capture(shape=(256,)),)
    )


def test_cache_key_includes_payload_target_and_algorithm_options():
    base = _key()
    assert base != _key(payload=b"different-ltoir")
    assert base != _key(target=_target(sm="100"))
    assert base != _key(algorithm_options={"adapter": "ivf_flat", "dtype": "int32"})


def test_make_udf_artifact_sets_cache_key_and_descriptor_metadata():
    artifact = make_udf_artifact(
        abi="rapids.cuvs.ivf_flat.metric.v1",
        payload_kind="ltoir",
        payload=b"fake-ltoir",
        symbol_name="cuvs_l2_update_f32",
        captures=(_capture(pointer=42),),
        target=_target(),
        source_hash=source_hash("def f(x, y, acc): return acc"),
        lowering_version="ctx-lowering-v1",
        algorithm_options={"adapter": "ivf_flat"},
    )

    assert artifact.cache_key.startswith("rapids.cuvs.device_udf.cache_key.v1:")
    assert artifact.payload_bytes() == b"fake-ltoir"

    metadata = artifact.c_descriptor_metadata()
    assert metadata["payload_size"] == len(b"fake-ltoir")
    assert metadata["captures"][0]["pointer"] == 42
    assert "owner" not in metadata["captures"][0]


def test_ivf_flat_cuda_source_metric_builds_cuda_source_artifact():
    from cuvs.neighbors import ivf_flat

    artifact = ivf_flat.cuda_source_metric(
        'extern "C" __device__ float f(float x, float y, float acc);',
        symbol_name="cuvs_test_source_metric",
    )

    assert artifact.abi == "rapids.cuvs.ivf_flat.metric.v1"
    assert artifact.payload_kind == "cuda_source"
    assert artifact.symbol_name == "cuvs_test_source_metric"
    assert artifact.payload_bytes().startswith(b'extern "C"')
    assert artifact.target.sm == "runtime"


def test_ivf_flat_search_params_accepts_cuda_source_udf_artifact():
    from cuvs.neighbors import ivf_flat

    artifact = _artifact(
        payload_kind="cuda_source",
        payload='extern "C" __device__ float f(float x, float y, float acc);',
    )

    params = ivf_flat.SearchParams(metric=artifact)

    assert params.metric is artifact


def test_ivf_flat_search_params_rejects_cuda_source_captures():
    from cuvs.neighbors import ivf_flat

    artifact = _artifact(
        payload_kind="cuda_source",
        payload='extern "C" __device__ float f(float x, float y, float acc);',
        captures=(_capture(name="weights", pointer=1234),),
    )

    with pytest.raises(NotImplementedError, match="do not support captures"):
        ivf_flat.SearchParams(metric=artifact)


def test_ivf_flat_search_params_rejects_wrong_abi():
    from cuvs.neighbors import ivf_flat

    artifact = _artifact(abi="rapids.cuvs.ivf_pq.metric.v1")

    with pytest.raises(ValueError, match="incompatible ABI"):
        ivf_flat.SearchParams(metric=artifact)


def test_ivf_flat_search_params_rejects_too_many_captures():
    from cuvs.neighbors import ivf_flat

    artifact = _artifact(
        captures=(
            _capture(name="weights", pointer=1234),
            _capture(name="bias", pointer=5678),
        )
    )

    with pytest.raises(NotImplementedError, match="at most one capture"):
        ivf_flat.SearchParams(metric=artifact)


def test_invalid_payload_kind_fails():
    with pytest.raises(ValueError, match="payload_kind"):
        _key(payload_kind="ptx")
