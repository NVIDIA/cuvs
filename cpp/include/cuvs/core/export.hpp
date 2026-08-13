/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Symbol visibility macros for cuVS shared libraries.
// When CXX_VISIBILITY_PRESET is set to hidden, only symbols explicitly
// marked with CUVS_EXPORT will be visible in the shared library.
// CUVS_HIDDEN can be used to explicitly mark symbols as hidden.
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define CUVS_EXPORT __attribute__((visibility("default")))
#define CUVS_HIDDEN __attribute__((visibility("hidden")))
#else
#define CUVS_EXPORT
#define CUVS_HIDDEN
#endif
