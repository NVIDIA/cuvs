/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Vamana: builds a DiskANN-style Vamana graph over a dataset.
//!
//! Build an [`Index`] from a dataset (then typically serialize it). The dataset
//! is borrowed through the `AsDlTensor` trait; see the [`dlpack`](crate::dlpack)
//! module for the tensor model.

mod index;
mod params;

pub use index::Index;
pub use params::IndexParams;

use crate::dlpack::DLPackError;
use crate::error::LibraryError;

/// Error type for Vamana operations.
#[derive(Debug, thiserror::Error)]
pub enum VamanaError {
    /// The cuVS C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
    /// A file path contained an interior NUL byte.
    #[error("path contains an interior NUL byte")]
    InvalidPath(#[from] std::ffi::NulError),
}
