# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped bootstrap for live PyLucene tests."""

from __future__ import annotations

import pytest

from cuvs_bench.tests.pylucene._pylucene_live_test_config import (
    PyLuceneRuntimeFixture,
    configure_pylucene_runtime,
)


@pytest.fixture(scope="session")
def pylucene_runtime_config(
    tmp_path_factory: pytest.TempPathFactory,
) -> PyLuceneRuntimeFixture:
    """Prepare the process-global JVM classpath before any live test runs."""
    return configure_pylucene_runtime(tmp_path_factory)
