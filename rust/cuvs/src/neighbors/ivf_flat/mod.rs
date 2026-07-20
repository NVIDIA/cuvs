/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IVF-Flat: an inverted-file index over uncompressed ("flat") vectors. It
//! partitions the dataset into `n_lists` clusters and, at query time, scans only
//! the `n_probes` closest clusters — a simple knob to trade recall for speed.
//!
//! Build an [`Index`] from a dataset, then [`search`](Index::search) it with
//! device-resident queries and output buffers. Tensors are borrowed through the
//! `AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
//! module for the tensor model and `examples/cagra.rs` for the same build/search
//! workflow.

mod index;
mod params;

pub use crate::neighbors::filters::SearchFilter;
pub use index::Index;
pub use params::{IndexParams, SearchParams};

use crate::dlpack::DLPackError;
use crate::error::LibraryError;

/// Error type for IVF-Flat operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IvfFlatError {
    /// The cuVS C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
    /// A parameter value failed validation.
    #[error("invalid parameter: {0}")]
    Validation(String),
}
