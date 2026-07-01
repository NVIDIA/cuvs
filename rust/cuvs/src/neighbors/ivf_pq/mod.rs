/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! IVF-PQ: an inverted-file index that product-quantizes the vectors. Like
//! IVF-Flat it partitions the dataset into `n_lists` clusters and scans the
//! `n_probes` closest at query time, but compresses each vector into `pq_dim`
//! codes of `pq_bits` bits — much smaller, slightly less accurate.
//!
//! Build an [`Index`] from a dataset, then [`search`](Index::search) it with
//! device-resident queries and output buffers. Tensors are borrowed through the
//! `AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
//! module for the tensor model and `examples/cagra.rs` for the same build/search
//! workflow.

mod index;
mod params;

pub use index::Index;
pub use params::{
    IndexParams, SearchParams, cudaDataType_t, cuvsIvfPqCodebookGen, cuvsIvfPqListLayout,
};

use crate::dlpack::DLPackError;
use crate::error::LibraryError;

/// Error type for IVF-PQ operations.
#[derive(Debug, thiserror::Error)]
pub enum IvfPqError {
    /// The cuVS C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
}
