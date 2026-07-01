/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Nearest neighbor search algorithms.
//!
//! Mirrors the C++ `cuvs::neighbors` namespace: each submodule wraps one index
//! type. Build an [`Index`](cagra::Index) from a dataset, then search it with
//! device-resident queries and output buffers; see the [`dlpack`](crate::dlpack)
//! module for the tensor model.

pub mod brute_force;
pub mod cagra;
pub mod ivf_flat;
pub mod ivf_pq;
pub mod vamana;
