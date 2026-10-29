/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include <cub/util_device.cuh>
#include <cuvs/core/cuda_fp16.hpp>

template cudaError_t cub::PtxVersionUncached<void>(int&);
