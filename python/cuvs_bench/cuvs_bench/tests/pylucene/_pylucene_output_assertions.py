#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Assertions for output emitted by PyLucene, the JVM, and native cuVS."""

_GRAPH_CLAMP_WARNING_FRAGMENTS = (
    "Intermediate graph degree cannot be larger",
    "cannot be larger than intermediate graph degree",
    "for nn-descent needs to match cagra intermediate graph degree",
)


def assert_no_cuvs_graph_clamp_warnings(captured_output: str) -> None:
    normalized_output = captured_output.lower()
    for warning_fragment in _GRAPH_CLAMP_WARNING_FRAGMENTS:
        assert warning_fragment.lower() not in normalized_output, (
            f"cuVS clamped the configured graph parameters:\n{captured_output}"
        )
