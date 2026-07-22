# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import keyword
import re
import textwrap
from collections.abc import Mapping
from typing import Any, Callable

from cuvs._lib.device_udf import (
    UDFArtifact,
    UDFCapture,
    UDFTarget,
    make_udf_artifact,
    source_hash,
)
from cuvs._lib.udf_validation import (
    CaptureInfo,
    UnsupportedUDFCapture,
    UDFCompilationError,
    validate_capture,
    validate_ivf_flat_metadata,
    validate_signature,
    validate_udf_policy,
)

_IVF_FLAT_METRIC_ABI = "rapids.cuvs.ivf_flat.metric.v1"
_LOWERING_VERSION = "ivf-flat-ctx-explicit-lowered-v1"
_CUDA_SOURCE_LOWERING_VERSION = "ivf-flat-cuda-source-v1"
_CAPTURE_NAME_RE = re.compile(r"^[A-Za-z_][0-9A-Za-z_]*$")


def cuda_source_metric(
    source: str | bytes,
    *,
    order: str = "min",
    initial: float = 0.0,
    coarse_metric: str = "sqeuclidean",
    symbol_name: str = "cuvs_py_ivf_flat_cuda_source_metric",
    target: UDFTarget | None = None,
) -> UDFArtifact:
    """Package an expert CUDA/C++ IVF-Flat metric source string.

    This is an explicit advanced path for source compatible with the
    existing C++ ``search_params.metric_udf`` contract. cuVS appends the
    IVF-Flat JIT adapter at search time, so the source must define
    ``cuvs::neighbors::ivf_flat::detail::compute_dist_udf_impl`` with
    the expected template signature.
    """

    validate_ivf_flat_metadata(order, initial, coarse_metric)
    if initial != 0.0:
        raise NotImplementedError(
            "ivf_flat.cuda_source_metric currently requires initial=0.0"
        )
    if not isinstance(source, (str, bytes)):
        raise TypeError("source must be str or bytes")
    if not _payload_bytes(source):
        raise ValueError("source must not be empty")
    if not symbol_name:
        raise ValueError("symbol_name must not be empty")

    return make_udf_artifact(
        abi=_IVF_FLAT_METRIC_ABI,
        payload_kind="cuda_source",
        payload=source,
        symbol_name=symbol_name,
        captures=(),
        target=target or _runtime_cuda_source_target(),
        source_hash=source_hash(source),
        lowering_version=_CUDA_SOURCE_LOWERING_VERSION,
        algorithm_options={
            "algorithm": "ivf_flat",
            "coarse_metric": coarse_metric,
            "capture_count": 0,
            "initial": initial,
            "order": order,
        },
    )


def metric(
    fn: Callable[..., Any] | None = None,
    *,
    order: str = "min",
    initial: float = 0.0,
    coarse_metric: str = "sqeuclidean",
    captures: Mapping[str, Any] | None = None,
    symbol_name: str | None = None,
    forceinline: bool = True,
) -> Callable[[Callable[..., Any]], UDFArtifact] | UDFArtifact:
    """Compile an IVF-Flat metric UDF to a device artifact.

    The decorator accepts the user-facing ``f(x, y, acc, ctx)`` shape and
    lowers it to the coordinate-wise IVF-Flat metric ABI. V1 supports no
    captures or one contiguous ``float32`` CUDA-array capture, addressable as
    ``ctx.<capture_name>[ctx.dim]``.
    """

    def decorate(user_fn: Callable[..., Any]) -> UDFArtifact:
        return _compile_metric(
            user_fn,
            order=order,
            initial=initial,
            coarse_metric=coarse_metric,
            captures=captures,
            symbol_name=symbol_name,
            forceinline=forceinline,
        )

    if fn is None:
        return decorate
    if not callable(fn):
        raise TypeError("ivf_flat.metric must decorate a callable")
    return decorate(fn)


def _compile_metric(
    fn: Callable[..., Any],
    *,
    order: str,
    initial: float,
    coarse_metric: str,
    captures: Mapping[str, Any] | None,
    symbol_name: str | None,
    forceinline: bool,
) -> UDFArtifact:
    validate_ivf_flat_metadata(order, initial, coarse_metric)
    if initial != 0.0:
        raise NotImplementedError(
            "ivf_flat.metric currently requires initial=0.0"
        )

    validate_signature(fn, ["x", "y", "acc", "ctx"])
    validate_udf_policy(fn)
    capture_infos = _validate_captures(captures)

    source = _function_source(fn)
    c_symbol = symbol_name or _default_symbol_name(fn, source)
    lowered_fn, lowered_source = _lower_metric(
        fn, c_symbol, source, capture_infos
    )

    from cuvs._lib.udf_backends.numba_cuda_mlir import (
        NumbaCudaMLIRBackend,
        current_target,
    )

    backend = NumbaCudaMLIRBackend()
    if not backend.cuda.is_available():
        raise UDFCompilationError("CUDA is not available to numba_cuda_mlir")

    _validate_capture_devices(capture_infos, _current_device_id(backend.cuda))
    udf_captures = tuple(_to_udf_capture(capture) for capture in capture_infos)
    arg_types = [
        backend.float32(),
        backend.float32(),
        backend.float32(),
    ]
    for capture in capture_infos:
        arg_types.append(_capture_arg_type(backend, capture))
        arg_types.append(backend.int64())

    target = current_target(backend.cuda, compile_options=("-lto",))
    result = backend.compile(
        lowered_fn,
        abi=_IVF_FLAT_METRIC_ABI,
        symbol_name=c_symbol,
        arg_types=tuple(arg_types),
        return_type=backend.float32(),
        target=target,
        captures=udf_captures,
        lowering_version=_LOWERING_VERSION,
        algorithm_options={
            "algorithm": "ivf_flat",
            "coarse_metric": coarse_metric,
            "capture_count": len(capture_infos),
            "initial": initial,
            "order": order,
        },
        forceinline=forceinline,
        source=lowered_source,
    )
    return result.artifact


def _lower_metric(
    fn: Callable[..., Any],
    symbol_name: str,
    source: str,
    captures: tuple[CaptureInfo, ...],
) -> tuple[Callable[..., Any], str]:
    tree = ast.parse(source)
    function_defs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(function_defs) != 1:
        raise TypeError(
            "expected exactly one IVF-Flat metric function definition"
        )

    func = function_defs[0]
    func.decorator_list = []
    if not captures and _uses_name(func, "ctx"):
        raise NotImplementedError(
            "ivf_flat.metric ctx access requires an explicit capture"
        )

    if captures:
        func = _CtxLoweringTransformer(
            {capture.name for capture in captures}
        ).visit(func)
        if _uses_name(func, "ctx"):
            raise NotImplementedError(
                "ivf_flat.metric ctx may only be used as ctx.dim or "
                "ctx.<capture>"
            )

    func.name = symbol_name
    args = [ast.arg(arg="x"), ast.arg(arg="y"), ast.arg(arg="acc")]
    for capture in captures:
        args.append(ast.arg(arg=capture.name))
        args.append(ast.arg(arg="dim"))
    func.args = ast.arguments(
        posonlyargs=[],
        args=args,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    tree.body = [func]
    ast.fix_missing_locations(tree)

    lowered_source = ast.unparse(tree)
    namespace = dict(fn.__globals__)
    exec(
        compile(tree, f"<cuvs-ivf-flat-metric:{symbol_name}>", "exec"),
        namespace,
    )
    lowered_fn = namespace[symbol_name]
    lowered_fn.__module__ = fn.__module__
    lowered_fn.__qualname__ = symbol_name
    return lowered_fn, lowered_source


def _validate_captures(
    captures: Mapping[str, Any] | None,
) -> tuple[CaptureInfo, ...]:
    if not captures:
        return ()
    if len(captures) > 1:
        raise NotImplementedError(
            "ivf_flat.metric currently supports at most one capture"
        )

    capture_infos = []
    for name, value in captures.items():
        if not isinstance(name, str) or not _CAPTURE_NAME_RE.match(name):
            raise TypeError("capture names must be valid Python identifiers")
        if keyword.iskeyword(name) or name in {"x", "y", "acc", "ctx", "dim"}:
            raise ValueError(f"capture name {name!r} is reserved")

        capture = validate_capture(name, value)
        if capture.dtype != "float32":
            raise NotImplementedError(
                "ivf_flat.metric currently supports only float32 captures"
            )
        if len(capture.shape) != 1:
            raise UnsupportedUDFCapture(
                "ivf_flat.metric captures must be one-dimensional"
            )
        if capture.strides not in (None, (4,)):
            raise UnsupportedUDFCapture(
                "ivf_flat.metric captures must be contiguous"
            )
        capture_infos.append(capture)
    return tuple(capture_infos)


def _validate_capture_devices(
    captures: tuple[CaptureInfo, ...], expected_device: int | None
) -> None:
    if expected_device is None:
        return
    for capture in captures:
        if capture.device_id != expected_device:
            raise UnsupportedUDFCapture(
                "capture device must match cuVS resource device"
            )


def _to_udf_capture(capture: CaptureInfo) -> UDFCapture:
    return UDFCapture(
        name=capture.name,
        dtype=capture.dtype,
        shape=capture.shape,
        strides=capture.strides,
        device_id=capture.device_id,
        readonly=capture.readonly,
        pointer=capture.pointer,
        owner=capture.owner,
    )


def _capture_arg_type(backend: Any, capture: CaptureInfo) -> Any:
    if capture.dtype == "float32":
        return backend.float32_pointer()
    raise NotImplementedError(
        f"unsupported ivf_flat.metric capture dtype {capture.dtype!r}"
    )


def _current_device_id(cuda: Any) -> int | None:
    try:
        device = cuda.get_current_device()
    except Exception:
        return None

    for attr in ("id", "device_id", "ordinal"):
        value = getattr(device, attr, None)
        if value is not None:
            return int(value)
    return 0


def _function_source(fn: Callable[..., Any]) -> str:
    return textwrap.dedent(inspect.getsource(fn))


def _default_symbol_name(fn: Callable[..., Any], source: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z_]+", "_", fn.__qualname__).strip("_")
    if not stem:
        stem = "metric"
    if stem[0].isdigit():
        stem = f"f_{stem}"
    return f"cuvs_py_ivf_flat_metric_{stem}_{source_hash(source)[:16]}"


def _runtime_cuda_source_target() -> UDFTarget:
    return UDFTarget(
        sm="runtime",
        cuda_version="runtime",
        nvrtc_version="runtime",
        nvjitlink_version="runtime",
        numba_cuda_mlir_version=None,
        compile_options=("nvrtc-lto",),
    )


def _payload_bytes(payload: str | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return payload.encode("utf-8")


def _uses_name(node: ast.AST, name: str) -> bool:
    finder = _NameUseFinder(name)
    finder.visit(node)
    return finder.found


class _NameUseFinder(ast.NodeVisitor):
    def __init__(self, name: str):
        self.name = name
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name:
            self.found = True


class _CtxLoweringTransformer(ast.NodeTransformer):
    def __init__(self, capture_names: set[str]):
        self.capture_names = capture_names

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "ctx":
            if node.attr == "dim":
                return ast.copy_location(
                    ast.Name(id="dim", ctx=node.ctx), node
                )
            if node.attr in self.capture_names:
                return ast.copy_location(
                    ast.Name(id=node.attr, ctx=node.ctx), node
                )
            raise UnsupportedUDFCapture(
                f"ctx.{node.attr} has no matching capture"
            )
        return node
