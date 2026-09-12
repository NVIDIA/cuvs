#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for native/JVM output assertions used by the PyLucene suite."""

import pytest

from cuvs_bench.tests.pylucene._pylucene_output_assertions import (
    assert_no_cuvs_graph_clamp_warnings,
)


@pytest.mark.parametrize(
    "warning",
    (
        "Intermediate graph degree cannot be larger",
        "cannot be larger than intermediate graph degree",
        "for nn-descent needs to match cagra intermediate graph degree",
    ),
    ids=(
        "intermediate-degree-vs-dataset",
        "graph-degree-vs-intermediate-degree",
        "nn-descent-degree-mismatch",
    ),
)
def test_graph_clamp_warning_is_rejected_case_insensitively(warning):
    with pytest.raises(AssertionError, match="clamped"):
        assert_no_cuvs_graph_clamp_warnings(f"WARNING: {warning.upper()}")


def test_warning_free_output_is_accepted():
    assert_no_cuvs_graph_clamp_warnings(
        "PASS [GPU CAGRA search] documents=100, searchWidth=32"
    )
