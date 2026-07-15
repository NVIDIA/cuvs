/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Vamana: builds a DiskANN-style Vamana graph over a dataset.
//!
//! Build an [`Index`] from a dataset (then typically serialize it). The dataset
//! is borrowed through the [`AsDlTensor`](crate::AsDlTensor) trait; see the
//! [`dlpack`](crate::dlpack) module for the tensor model.

mod index;
mod index_params;

pub use index::Index;
pub use index_params::IndexParams;
