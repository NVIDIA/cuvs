/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::tiered_index::AnnAlgo;

/// Search parameters for a [`crate::tiered_index::Index`].
///
/// The variant must match the ANN backend the index was built with. The C API
/// (`cuvsTieredIndexSearch`) takes the backend's search params as an opaque
/// `void*` and reinterprets it according to the index's build-time algorithm —
/// passing the wrong variant would reinterpret the pointer as the wrong struct
/// (undefined behavior). [`crate::tiered_index::Index::search`] guards against
/// this by validating the variant against the index's algo and returning
/// [`crate::error::Error::InvalidArgument`] on mismatch.
#[derive(Debug)]
pub enum SearchParams {
    /// Search params for a CAGRA-backed tiered index.
    Cagra(crate::cagra::SearchParams),
    /// Search params for an IVF-Flat-backed tiered index.
    IvfFlat(crate::ivf_flat::SearchParams),
    /// Search params for an IVF-PQ-backed tiered index.
    IvfPq(crate::ivf_pq::SearchParams),
}

impl SearchParams {
    /// The ANN backend this variant targets, for validation against the index's
    /// build-time algorithm.
    pub(crate) fn algo(&self) -> AnnAlgo {
        match self {
            SearchParams::Cagra(_) => AnnAlgo::CUVS_TIERED_INDEX_ALGO_CAGRA,
            SearchParams::IvfFlat(_) => AnnAlgo::CUVS_TIERED_INDEX_ALGO_IVF_FLAT,
            SearchParams::IvfPq(_) => AnnAlgo::CUVS_TIERED_INDEX_ALGO_IVF_PQ,
        }
    }

    /// Raw pointer to the backend search params struct, type-erased to the
    /// opaque `void*` the C API expects. The caller must ensure the variant
    /// matches the index's build-time algorithm.
    pub(crate) fn as_void_ptr(&self) -> *mut std::os::raw::c_void {
        match self {
            SearchParams::Cagra(p) => p.0 as *mut _,
            SearchParams::IvfFlat(p) => p.0 as *mut _,
            SearchParams::IvfPq(p) => p.0 as *mut _,
        }
    }
}
