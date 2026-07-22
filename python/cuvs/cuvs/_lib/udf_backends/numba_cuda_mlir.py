# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cuvs._lib.device_udf import (
    UDFArtifact,
    UDFCapture,
    UDFTarget,
    make_udf_artifact,
    source_hash,
)
from cuvs._lib.udf_validation import UDFCompilationError


@dataclass(frozen=True)
class NumbaCudaMLIRCompileResult:
    artifact: UDFArtifact
    return_type: Any


class NumbaCudaMLIRBackend:
    payload_kind = "ltoir"

    def __init__(self, cuda_module: Any | None = None, types_module: Any | None = None):
        if cuda_module is None or types_module is None:
            from numba_cuda_mlir import cuda
            from numba_cuda_mlir.numba_cuda import types

            cuda_module = cuda if cuda_module is None else cuda_module
            types_module = types if types_module is None else types_module

        self.cuda = cuda_module
        self.types = types_module

    def compile(
        self,
        fn: Any,
        *,
        abi: str,
        symbol_name: str,
        arg_types: Sequence[Any],
        return_type: Any,
        target: UDFTarget,
        captures: tuple[UDFCapture, ...] = (),
        lowering_version: str,
        algorithm_options: Mapping[str, Any] | None = None,
        forceinline: bool = True,
        source: str | bytes | None = None,
    ) -> NumbaCudaMLIRCompileResult:
        sig = return_type(*arg_types)
        try:
            payload, actual_return_type = self.cuda.compile(
                fn,
                sig=sig,
                device=True,
                abi="c",
                abi_info={"abi_name": symbol_name},
                output="ltoir",
                forceinline=forceinline,
            )
        except Exception as exc:  # pragma: no cover - exercised in GPU env tests
            raise UDFCompilationError(str(exc)) from exc

        if actual_return_type != return_type:
            raise TypeError(
                f"UDF return type must be {return_type}, got {actual_return_type}"
            )

        artifact = make_udf_artifact(
            abi=abi,
            payload_kind="ltoir",
            payload=payload,
            symbol_name=symbol_name,
            captures=captures,
            target=target,
            source_hash=source_hash(
                source if source is not None else _function_source(fn)
            ),
            lowering_version=lowering_version,
            algorithm_options=algorithm_options,
        )
        return NumbaCudaMLIRCompileResult(
            artifact=artifact,
            return_type=actual_return_type,
        )

    def float32(self) -> Any:
        return self.types.float32

    def int64(self) -> Any:
        return self.types.int64

    def float32_pointer(self) -> Any:
        return self.types.CPointer(self.types.float32)


def current_target(cuda_module: Any, *, compile_options: Sequence[str] = ()) -> UDFTarget:
    dev = cuda_module.get_current_device()
    cc = getattr(dev, "compute_capability")
    sm = f"{cc.major}{cc.minor}"

    cuda_version = _version_or_unknown(cuda_module)
    try:
        import numba_cuda_mlir

        numba_cuda_mlir_version = getattr(numba_cuda_mlir, "__version__", None)
    except Exception:  # pragma: no cover
        numba_cuda_mlir_version = None

    return UDFTarget(
        sm=sm,
        cuda_version=cuda_version,
        nvrtc_version=None,
        nvjitlink_version="unknown",
        numba_cuda_mlir_version=numba_cuda_mlir_version,
        compile_options=tuple(compile_options),
    )


def _function_source(fn: Any) -> str:
    try:
        return textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return getattr(fn, "__qualname__", repr(fn))


def _version_or_unknown(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))
