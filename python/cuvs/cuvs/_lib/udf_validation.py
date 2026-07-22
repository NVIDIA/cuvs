# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class UnsupportedUDFSyntax(TypeError):
    pass


class UnsupportedUDFCapture(TypeError):
    pass


class UDFCompilationError(RuntimeError):
    pass


class UDFABIError(RuntimeError):
    pass


ALLOWED_CALLS = frozenset(
    {
        "abs",
        "min",
        "max",
        "squared_diff",
        "abs_diff",
        "dot_product",
        "range",
    }
)
_ALLOWED_NODE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Load,
    ast.Store,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.Expr,
    ast.Name,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.If,
    ast.IfExp,
    ast.For,
    ast.Call,
    ast.Subscript,
    ast.Attribute,
    ast.Tuple,
    ast.List,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def validate_signature(fn: Any, expected: Iterable[str]) -> None:
    expected_names = list(expected)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    names = [p.name for p in params]

    if names != expected_names:
        raise TypeError(f"expected f({', '.join(expected_names)})")

    for param in params:
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError("*args, **kwargs, and keyword-only args are not allowed")

        if param.default is not inspect.Parameter.empty:
            raise TypeError("UDF parameters may not have defaults")


def validate_ivf_flat_metadata(order: str, initial: Any, coarse_metric: str) -> None:
    if order not in {"min"}:
        raise ValueError("v1 supports order='min'")

    if not isinstance(initial, (int, float)):
        raise TypeError("initial must be a scalar number")

    if coarse_metric not in {"sqeuclidean", "inner_product"}:
        raise ValueError("coarse_metric must be explicit and supported")


def validate_udf_policy(fn: Any) -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != 1:
        raise UnsupportedUDFSyntax("expected exactly one UDF function definition")

    for function_def in function_defs:
        function_def.decorator_list = []

    UDFPolicyValidator().visit(tree)
    return function_defs[0]


class UDFPolicyValidator(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise UnsupportedUDFSyntax(f"unsupported syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        raise UnsupportedUDFSyntax("imports are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise UnsupportedUDFSyntax("imports are not allowed")

    def visit_While(self, node: ast.While) -> None:
        raise UnsupportedUDFSyntax("while loops are not allowed in v1")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise UnsupportedUDFSyntax("lambda expressions are not allowed")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise UnsupportedUDFSyntax("comprehensions are not allowed")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise UnsupportedUDFSyntax("comprehensions are not allowed")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise UnsupportedUDFSyntax("comprehensions are not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise UnsupportedUDFSyntax("comprehensions are not allowed")

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node)
        if name not in ALLOWED_CALLS:
            raise UnsupportedUDFSyntax(f"call to {name} is not allowed")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if is_ctx_capture_write(target):
                raise UnsupportedUDFSyntax("captures are read-only")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if is_ctx_capture_write(node.target):
            raise UnsupportedUDFSyntax("captures are read-only")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if is_ctx_capture_write(node.target):
            raise UnsupportedUDFSyntax("captures are read-only")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if not _is_range_call(node.iter):
            raise UnsupportedUDFSyntax("for loops must iterate over range(...) in v1")
        self.generic_visit(node)


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = []
        current: ast.AST = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return type(node.func).__name__


def is_ctx_capture_write(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "ctx"
    if isinstance(node, ast.Subscript):
        return _contains_ctx_attribute(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(is_ctx_capture_write(elt) for elt in node.elts)
    return False


def _contains_ctx_attribute(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "ctx"
    if isinstance(node, ast.Subscript):
        return _contains_ctx_attribute(node.value)
    return False


def _is_range_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and call_name(node) == "range"


@dataclass(frozen=True)
class CaptureInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    device_id: int
    pointer: int
    owner: object
    readonly: bool = True


_SUPPORTED_CAPTURE_DTYPES = frozenset({"float32", "int32", "int64"})
_TYPES_BY_CAI_TYPESTR = {
    "<f4": "float32",
    "|f4": "float32",
    "=f4": "float32",
    "<i4": "int32",
    "|i4": "int32",
    "=i4": "int32",
    "<i8": "int64",
    "|i8": "int64",
    "=i8": "int64",
}


def validate_capture(name: str, value: Any, expected_device: int | None = None) -> CaptureInfo:
    cai = getattr(value, "__cuda_array_interface__", None)
    if cai is None:
        raise UnsupportedUDFCapture(f"capture {name!r} must expose CUDA Array Interface")

    typestr = cai.get("typestr")
    dtype = _TYPES_BY_CAI_TYPESTR.get(typestr)
    if dtype not in _SUPPORTED_CAPTURE_DTYPES:
        raise UnsupportedUDFCapture(f"unsupported capture dtype: {typestr}")

    data = cai.get("data")
    if not isinstance(data, tuple) or len(data) < 1:
        raise UnsupportedUDFCapture(f"capture {name!r} has invalid CUDA Array Interface data")

    pointer = int(data[0])
    # Most device-array providers, including CuPy, expose mutable arrays via
    # CUDA Array Interface. cuVS treats captures as read-only by policy: UDF
    # validation rejects writes, and generated wrappers pass const pointers.
    readonly = True

    shape = tuple(int(dim) for dim in cai.get("shape", ()))
    strides_raw = cai.get("strides")
    strides = None if strides_raw is None else tuple(int(stride) for stride in strides_raw)

    device_id = _capture_device_id(cai)
    if expected_device is not None and device_id != expected_device:
        raise UnsupportedUDFCapture("capture device must match cuVS resource device")

    return CaptureInfo(
        name=name,
        dtype=dtype,
        shape=shape,
        strides=strides,
        device_id=device_id,
        pointer=pointer,
        owner=value,
        readonly=readonly,
    )


def _capture_device_id(cai: dict[str, Any]) -> int:
    stream = cai.get("stream")
    if isinstance(stream, tuple) and len(stream) >= 1 and isinstance(stream[0], int):
        return stream[0]
    return int(cai.get("device", 0))
