/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CAGRA: a graph-based approximate nearest neighbors algorithm with
//! state-of-the-art query throughput for both small and large batch sizes.
//!
//! Build an [`Index`] from a dataset, then [`search`](Index::search) it with
//! device-resident queries and output buffers. Tensors are passed through the
//! `AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
//! module for the tensor model and `examples/cagra.rs` for a complete, runnable
//! example.
//!
//! Parameter types ([`IndexParams`], [`SearchParams`], ...) use the [`bon`]
//! builder pattern: every setter is optional and unset values keep the cuVS C
//! library defaults. Values are validated when the builder's `build()` runs,
//! returning [`CagraError::Validation`] for out-of-range inputs.

mod index;
mod params;

pub use index::Index;
pub use params::{CompressionParams, IndexParams, SearchParams};

use crate::dlpack::DLPackError;
use crate::error::LibraryError;

/// Algorithm for building the internal k-NN graph.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphBuildAlgo {
    /// Automatically select the best algorithm.
    Auto,
    /// Build using IVF-PQ.
    IvfPq,
    /// Build using NN-Descent.
    NnDescent,
    /// Build using iterative CAGRA search.
    IterativeCagraSearch,
    /// Build using ACE (Augmented Core Extraction) for large datasets.
    Ace,
}

impl From<GraphBuildAlgo> for ffi::cuvsCagraGraphBuildAlgo {
    fn from(v: GraphBuildAlgo) -> Self {
        match v {
            GraphBuildAlgo::Auto => Self::AUTO_SELECT,
            GraphBuildAlgo::IvfPq => Self::IVF_PQ,
            GraphBuildAlgo::NnDescent => Self::NN_DESCENT,
            GraphBuildAlgo::IterativeCagraSearch => Self::ITERATIVE_CAGRA_SEARCH,
            GraphBuildAlgo::Ace => Self::ACE,
        }
    }
}

impl From<ffi::cuvsCagraGraphBuildAlgo> for GraphBuildAlgo {
    fn from(v: ffi::cuvsCagraGraphBuildAlgo) -> Self {
        match v {
            ffi::cuvsCagraGraphBuildAlgo::AUTO_SELECT => Self::Auto,
            ffi::cuvsCagraGraphBuildAlgo::IVF_PQ => Self::IvfPq,
            ffi::cuvsCagraGraphBuildAlgo::NN_DESCENT => Self::NnDescent,
            ffi::cuvsCagraGraphBuildAlgo::ITERATIVE_CAGRA_SEARCH => Self::IterativeCagraSearch,
            ffi::cuvsCagraGraphBuildAlgo::ACE => Self::Ace,
        }
    }
}

/// Search kernel implementation.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchAlgo {
    /// Single CTA -- best for large batch sizes.
    SingleCta,
    /// Multi CTA -- best for small batch sizes.
    MultiCta,
    /// Multi kernel -- best for small batch sizes.
    MultiKernel,
    /// Automatically select the best kernel.
    Auto,
}

impl From<SearchAlgo> for ffi::cuvsCagraSearchAlgo {
    fn from(v: SearchAlgo) -> Self {
        match v {
            SearchAlgo::SingleCta => Self::SINGLE_CTA,
            SearchAlgo::MultiCta => Self::MULTI_CTA,
            SearchAlgo::MultiKernel => Self::MULTI_KERNEL,
            SearchAlgo::Auto => Self::AUTO,
        }
    }
}

impl From<ffi::cuvsCagraSearchAlgo> for SearchAlgo {
    fn from(v: ffi::cuvsCagraSearchAlgo) -> Self {
        match v {
            ffi::cuvsCagraSearchAlgo::SINGLE_CTA => Self::SingleCta,
            ffi::cuvsCagraSearchAlgo::MULTI_CTA => Self::MultiCta,
            ffi::cuvsCagraSearchAlgo::MULTI_KERNEL => Self::MultiKernel,
            ffi::cuvsCagraSearchAlgo::AUTO => Self::Auto,
        }
    }
}

/// Hash-table mode used during search.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum HashMode {
    /// Standard hash table.
    Hash,
    /// Small hash table optimised for low memory.
    Small,
    /// Automatically select the best mode.
    Auto,
}

impl From<HashMode> for ffi::cuvsCagraHashMode {
    fn from(v: HashMode) -> Self {
        match v {
            HashMode::Hash => Self::HASH,
            HashMode::Small => Self::SMALL,
            HashMode::Auto => Self::AUTO_HASH,
        }
    }
}

impl From<ffi::cuvsCagraHashMode> for HashMode {
    fn from(v: ffi::cuvsCagraHashMode) -> Self {
        match v {
            ffi::cuvsCagraHashMode::HASH => Self::Hash,
            ffi::cuvsCagraHashMode::SMALL => Self::Small,
            ffi::cuvsCagraHashMode::AUTO_HASH => Self::Auto,
        }
    }
}

/// Error type for CAGRA operations.
#[derive(Debug, thiserror::Error)]
pub enum CagraError {
    /// The cuVS C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
    /// A file path contained an interior NUL byte.
    #[error("path contains an interior NUL byte")]
    InvalidPath(#[from] std::ffi::NulError),
    /// A parameter value failed validation.
    #[error("invalid parameter: {0}")]
    Validation(String),
}
