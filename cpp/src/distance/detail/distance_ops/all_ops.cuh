/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Defines a named requirement "has_cutlass_op"
#include "cutlass.cuh"

// The distance operations:
#include "../distance_ops/bitwise_hamming.cuh"
#include "../distance_ops/canberra.cuh"
#include "../distance_ops/correlation.cuh"
#include "../distance_ops/cosine.cuh"
#include "../distance_ops/hamming.cuh"
#include "../distance_ops/hellinger.cuh"
#include "../distance_ops/jensen_shannon.cuh"
#include "../distance_ops/kl_divergence.cuh"
#include "../distance_ops/l1.cuh"
#include "../distance_ops/l2_exp.cuh"
#include "../distance_ops/l2_unexp.cuh"
#include "../distance_ops/l_inf.cuh"
#include "../distance_ops/lp_unexp.cuh"
#include "../distance_ops/russel_rao.cuh"
