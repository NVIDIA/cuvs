#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Validation helpers for names used in benchmark artifacts."""

from __future__ import annotations

import os
import re
from typing import Any


_RESULT_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")


def validate_path_component(value: Any, description: str) -> str:
    """Return a safe, non-empty filesystem path component."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or os.path.basename(value) != value
        or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        )
    ):
        raise ValueError(f"Invalid {description}: {value!r}")
    return value


def validate_result_component(value: Any, description: str) -> str:
    """Return a path component safe for comma-delimited result identities."""
    component = validate_path_component(value, description)
    if _RESULT_COMPONENT.fullmatch(component) is None:
        raise ValueError(f"Invalid {description}: {value!r}")
    return component


def format_artifact_identity(
    algorithm: Any,
    group: Any,
    scope: Any = None,
    *,
    description: str = "benchmark",
) -> str:
    """Format an injective identity from validated artifact components."""
    algorithm = validate_result_component(
        algorithm, f"{description} algorithm name"
    )
    group = validate_result_component(group, f"{description} group name")
    if scope is not None:
        scope = validate_result_component(scope, f"{description} result scope")

    identity = algorithm
    if group != "base":
        identity += f"[group={group}]"
    if scope is not None:
        identity += f"[scope={scope}]"
    return identity
