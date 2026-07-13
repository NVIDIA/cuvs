#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Pytest configuration for cuvs_bench tests."""


def pytest_configure(config):
    """Register elastic plugin when elasticsearch is available.

    Ensures elastic tests run when elasticsearch is installed, even if
    cuvs-bench-elastic was not installed via pip (e.g. using PYTHONPATH).
    """
    try:
        import elasticsearch  # noqa: F401
    except ImportError:
        return

    try:
        from cuvs_bench_elastic import register
    except ImportError:
        from cuvs_bench.backends.elasticsearch import register

    register()
