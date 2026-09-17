#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Pytest configuration for cuvs_bench tests."""

import pytest


def pytest_addoption(parser):
    """Add the explicit opt-in for live Lucene integration tests."""
    parser.addoption(
        "--run-lucene-e2e",
        action="store_true",
        default=False,
        help=(
            "run live Lucene tests; all cases require PyLucene/Java, and "
            "CAGRA cases additionally require cuVS/GPU"
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Skip live Lucene cases unless the suite was selected explicitly."""
    if config.getoption("--run-lucene-e2e"):
        return
    skip = pytest.mark.skip(reason="requires --run-lucene-e2e")
    for item in items:
        if "lucene_e2e" in item.keywords:
            item.add_marker(skip)


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
