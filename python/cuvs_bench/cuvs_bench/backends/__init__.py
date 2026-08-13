#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
cuvs-bench backends package.

This package provides the plugin architecture for benchmarking various
vector database backends, including C++ executables, Python libraries,
and network-based VDB services.
"""

from .base import (
    Dataset,
    BuildResult,
    SearchResult,
    BenchmarkBackend,
)

from .registry import (
    BackendRegistry,
    get_registry,
    register_backend,
    get_backend,
)

from .cpp_gbench import CppGoogleBenchmarkBackend
from .opensearch import OpenSearchBackend

# Auto-register built-in backends
_registry = get_registry()
_registry.register("cpp_gbench", CppGoogleBenchmarkBackend)
_registry.register("opensearch", OpenSearchBackend)

__all__ = [
    # Base classes and data structures
    "Dataset",
    "BuildResult",
    "SearchResult",
    "BenchmarkBackend",
    # Registry
    "BackendRegistry",
    "get_registry",
    "register_backend",
    "get_backend",
    # Built-in backends
    "CppGoogleBenchmarkBackend",
    "OpenSearchBackend",
]
