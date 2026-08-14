/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * Minimal bindgen-only CUDA runtime stub.
 *
 * cuVS C headers currently include <cuda_runtime.h> only for cudaStream_t and
 * cudaDataType_t in public C ABI declarations. The Rust bindings provide their
 * own ABI-compatible definitions and blocklist CUDA items, so bindgen only needs
 * these declarations to parse the headers without having to discover a CUDA Toolkit.
 */
typedef struct CUstream_st* cudaStream_t;
typedef unsigned int cudaDataType_t;
