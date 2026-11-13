#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared serialization contract for configured and measured search plans."""

import json

import numpy as np


SEARCH_PARAMETER_ERROR = "_search_parameter_error"


class SearchParameterError(TypeError):
    """A search plan cannot be represented in benchmark result identities."""


def search_parameter_failure_message(results):
    """Summarize only parameter validation failures, not backend policy."""
    errors = [
        result.metadata[SEARCH_PARAMETER_ERROR]
        for result in results
        if SEARCH_PARAMETER_ERROR in result.metadata
    ]
    return "; ".join(errors) if errors else None


def search_params_identity(search_params, *, context="search_params"):
    """Validate parameters and return their stable, backwards-compatible key.

    Native JSON values and NumPy boolean, integer, and native-width floating
    scalars are supported, including nested values. Unsupported values are
    rejected rather than stringified or converted with loss of precision.
    """
    try:
        return json.dumps(
            search_params,
            sort_keys=True,
            separators=(",", ":"),
            default=_numpy_parameter_as_python_scalar,
        )
    except (TypeError, ValueError) as error:
        raise SearchParameterError(f"{context}: {error}") from error


def _numpy_parameter_as_python_scalar(value):
    """Convert only NumPy scalars with an equivalent native JSON scalar."""
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        native_value = value.item()
        # Extended floats cannot be converted losslessly; time values are
        # excluded by dtype kind even though timedelta64 subclasses integer.
        if type(native_value) in (bool, int, float):
            return native_value
    raise TypeError(
        f"Unsupported search parameter type: {type(value).__name__}"
    )
