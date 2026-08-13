/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Wrapper for cuda_fp16.h that ensures __half gets default symbol visibility.
//
// GCC's "type visibility" rule causes template instantiations over __half to
// inherit hidden visibility when -fvisibility=hidden is in effect, because
// __half is a user-defined type first seen under hidden visibility. By
// including cuda_fp16.h under #pragma GCC visibility push(default), the __half
// type acquires default visibility, and downstream template instantiations
// (e.g., index<__half, ...>) will be properly exported from shared libraries.
#pragma GCC visibility push(default)
#include <cuda_fp16.h>  // NOLINT
#pragma GCC visibility pop
