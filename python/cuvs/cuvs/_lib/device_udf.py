# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

PayloadKind = Literal["ltoir", "cuda_source"]

_SUPPORTED_PAYLOAD_KINDS = frozenset({"ltoir", "cuda_source"})
_CACHE_KEY_VERSION = "rapids.cuvs.device_udf.cache_key.v1"


@dataclass(frozen=True)
class UDFTarget:
    sm: str
    cuda_version: str
    nvrtc_version: str | None
    nvjitlink_version: str
    numba_cuda_mlir_version: str | None
    compile_options: tuple[str, ...] = ()

    def cache_metadata(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class UDFCapture:
    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    device_id: int
    readonly: bool
    pointer: int
    owner: object = field(compare=False, hash=False, repr=False)

    def cache_metadata(self) -> dict[str, Any]:
        # Deliberately exclude pointer and owner. Pointers vary run to run; owner
        # only preserves lifetime and must not affect code identity.
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "strides": self.strides,
            "device_id": self.device_id,
            "readonly": self.readonly,
        }

    def c_descriptor_metadata(self) -> dict[str, Any]:
        return {
            **self.cache_metadata(),
            "pointer": self.pointer,
        }


@dataclass(frozen=True)
class UDFArtifact:
    abi: str
    payload_kind: PayloadKind
    payload: bytes | str
    symbol_name: str
    captures: tuple[UDFCapture, ...]
    target: UDFTarget
    cache_key: str

    def payload_bytes(self) -> bytes:
        return _payload_bytes(self.payload)

    def c_descriptor_metadata(self) -> dict[str, Any]:
        return {
            "abi": self.abi,
            "payload_kind": self.payload_kind,
            "payload_size": len(self.payload_bytes()),
            "symbol_name": self.symbol_name,
            "captures": [cap.c_descriptor_metadata() for cap in self.captures],
            "cache_key": self.cache_key,
        }


def make_udf_artifact(
    *,
    abi: str,
    payload_kind: PayloadKind,
    payload: bytes | str,
    symbol_name: str,
    captures: tuple[UDFCapture, ...] = (),
    target: UDFTarget,
    source_hash: str,
    lowering_version: str,
    algorithm_options: Mapping[str, Any] | None = None,
) -> UDFArtifact:
    cache_key = build_cache_key(
        abi=abi,
        payload_kind=payload_kind,
        payload=payload,
        target=target,
        captures=captures,
        source_hash=source_hash,
        lowering_version=lowering_version,
        algorithm_options=algorithm_options,
    )
    return UDFArtifact(
        abi=abi,
        payload_kind=payload_kind,
        payload=payload,
        symbol_name=symbol_name,
        captures=captures,
        target=target,
        cache_key=cache_key,
    )


def build_cache_key(
    *,
    abi: str,
    payload_kind: PayloadKind,
    payload: bytes | str,
    target: UDFTarget,
    captures: tuple[UDFCapture, ...] = (),
    source_hash: str,
    lowering_version: str,
    algorithm_options: Mapping[str, Any] | None = None,
) -> str:
    _validate_payload_kind(payload_kind)
    key_material = {
        "version": _CACHE_KEY_VERSION,
        "abi": abi,
        "payload_kind": payload_kind,
        "payload_hash": _sha256_hex(_payload_bytes(payload)),
        "target": target.cache_metadata(),
        "captures": [capture.cache_metadata() for capture in captures],
        "source_hash": source_hash,
        "lowering_version": lowering_version,
        "algorithm_options": dict(algorithm_options or {}),
    }
    return f"{_CACHE_KEY_VERSION}:{_sha256_hex(_stable_json_bytes(key_material))}"


def source_hash(source: str | bytes) -> str:
    return _sha256_hex(_payload_bytes(source))


def _validate_payload_kind(payload_kind: str) -> None:
    if payload_kind not in _SUPPORTED_PAYLOAD_KINDS:
        kinds = ", ".join(sorted(_SUPPORTED_PAYLOAD_KINDS))
        raise ValueError(f"payload_kind must be one of: {kinds}")


def _payload_bytes(payload: bytes | str) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    raise TypeError("payload must be bytes or str")


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
