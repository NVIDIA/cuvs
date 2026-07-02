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
pub use params::{IndexParams, SearchParams};

use crate::dlpack::DLPackError;
use crate::error::LibraryError;

/// Strategy for creating PQ codebooks.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodebookGen {
    /// One codebook per PQ subspace.
    PerSubspace,
    /// One codebook per IVF cluster.
    PerCluster,
}

impl From<CodebookGen> for ffi::cuvsIvfPqCodebookGen {
    fn from(v: CodebookGen) -> Self {
        match v {
            CodebookGen::PerSubspace => Self::CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE,
            CodebookGen::PerCluster => Self::CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER,
        }
    }
}

impl From<ffi::cuvsIvfPqCodebookGen> for CodebookGen {
    fn from(v: ffi::cuvsIvfPqCodebookGen) -> Self {
        match v {
            ffi::cuvsIvfPqCodebookGen::CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE => Self::PerSubspace,
            ffi::cuvsIvfPqCodebookGen::CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER => Self::PerCluster,
        }
    }
}

/// Memory layout of the IVF-PQ list data.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum ListLayout {
    /// Codes stored contiguously, one vector's codes after another.
    Flat,
    /// Codes interleaved for optimized search performance (the default).
    Interleaved,
}

impl From<ListLayout> for ffi::cuvsIvfPqListLayout {
    fn from(v: ListLayout) -> Self {
        match v {
            ListLayout::Flat => Self::CUVS_IVF_PQ_LIST_LAYOUT_FLAT,
            ListLayout::Interleaved => Self::CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED,
        }
    }
}

impl From<ffi::cuvsIvfPqListLayout> for ListLayout {
    fn from(v: ffi::cuvsIvfPqListLayout) -> Self {
        match v {
            ffi::cuvsIvfPqListLayout::CUVS_IVF_PQ_LIST_LAYOUT_FLAT => Self::Flat,
            ffi::cuvsIvfPqListLayout::CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED => Self::Interleaved,
        }
    }
}

/// Lookup-table dtype used during IVF-PQ search.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum LutDType {
    /// 32-bit floating-point lookup tables.
    F32,
    /// 16-bit floating-point lookup tables.
    F16,
    /// 8-bit unsigned lookup tables.
    U8,
}

impl From<LutDType> for ffi::cudaDataType_t {
    fn from(v: LutDType) -> Self {
        match v {
            LutDType::F32 => Self::CUDA_R_32F,
            LutDType::F16 => Self::CUDA_R_16F,
            LutDType::U8 => Self::CUDA_R_8U,
        }
    }
}

/// Accumulator dtype used for internal IVF-PQ distance computation.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum InternalDistanceDType {
    /// 32-bit floating-point accumulators.
    F32,
    /// 16-bit floating-point accumulators.
    F16,
}

impl From<InternalDistanceDType> for ffi::cudaDataType_t {
    fn from(v: InternalDistanceDType) -> Self {
        match v {
            InternalDistanceDType::F32 => Self::CUDA_R_32F,
            InternalDistanceDType::F16 => Self::CUDA_R_16F,
        }
    }
}

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
