/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::cagra::IndexParams as CagraIndexParams;
use crate::distance_type::DistanceType;
use crate::error::{Result, check_cuvs};
use crate::ivf_flat::IndexParams as IvfFlatIndexParams;
use crate::ivf_pq::IndexParams as IvfPqIndexParams;
use std::fmt;
use std::io::{Write, stderr};

/// Which ANN algorithm backs the tiered index's ANN tier.
pub type AnnAlgo = ffi::cuvsTieredIndexANNAlgo;

/// Supplemental parameters to build a [`crate::tiered_index::Index`].
///
/// A tiered index couples a brute-force tier (which absorbs incremental
/// inserts via [`crate::tiered_index::Index::extend`]) with an ANN tier. The
/// ANN tier is built once the incremental tier accumulates at least
/// `min_ann_rows` rows. Use [`IndexParams::set_algo`] to select which ANN
/// algorithm backs that tier (CAGRA by default), and the matching
/// `set_*_params` setter to supply per-algorithm build parameters.
///
/// The embedded ANN parameter wrappers are retained inside `IndexParams` so
/// their underlying C structs outlive the borrow taken by the tiered params
/// (which only stores raw pointers to them).
pub struct IndexParams {
    inner: ffi::cuvsTieredIndexParams_t,
    // Retain ownership of any embedded ANN params so they are not dropped while
    // the tiered params struct still points at them.
    cagra_params: Option<CagraIndexParams>,
    ivf_flat_params: Option<IvfFlatIndexParams>,
    ivf_pq_params: Option<IvfPqIndexParams>,
}

impl IndexParams {
    /// Returns a new IndexParams populated with cuVS defaults.
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsTieredIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsTieredIndexParamsCreate(params.as_mut_ptr()))?;
            Ok(IndexParams {
                inner: params.assume_init(),
                cagra_params: None,
                ivf_flat_params: None,
                ivf_pq_params: None,
            })
        }
    }

    /// Raw pointer to the underlying C params struct.
    pub(crate) fn as_ptr(&self) -> ffi::cuvsTieredIndexParams_t {
        self.inner
    }

    /// Which ANN algorithm backs the ANN tier, captured at build time so the
    /// index can validate that search params match the backend.
    pub(crate) fn algo(&self) -> AnnAlgo {
        unsafe { (*self.inner).algo }
    }

    /// DistanceType to use for building the index.
    pub fn set_metric(self, metric: DistanceType) -> IndexParams {
        unsafe {
            (*self.inner).metric = metric;
        }
        self
    }

    /// Which ANN algorithm backs the ANN tier (CAGRA, IVF-Flat, or IVF-PQ).
    pub fn set_algo(self, algo: AnnAlgo) -> IndexParams {
        unsafe {
            (*self.inner).algo = algo;
        }
        self
    }

    /// The minimum number of rows necessary in the index before an ANN index
    /// is created. Below this threshold, all rows live in the brute-force tier.
    pub fn set_min_ann_rows(self, min_ann_rows: i64) -> IndexParams {
        unsafe {
            (*self.inner).min_ann_rows = min_ann_rows;
        }
        self
    }

    /// Whether to (re)build the ANN tier on [`crate::tiered_index::Index::extend`]
    /// once the incremental (brute-force) tier exceeds `min_ann_rows`.
    pub fn set_create_ann_index_on_extend(self, create: bool) -> IndexParams {
        unsafe {
            (*self.inner).create_ann_index_on_extend = create;
        }
        self
    }

    /// Supply CAGRA build parameters for the ANN tier.
    ///
    /// Ownership of `params` is moved into `self` so the underlying C struct
    /// outlives the raw pointer stored in the tiered params.
    pub fn set_cagra_params(mut self, params: CagraIndexParams) -> IndexParams {
        unsafe {
            (*self.inner).cagra_params = params.0;
        }
        self.cagra_params = Some(params);
        self
    }

    /// Supply IVF-Flat build parameters for the ANN tier.
    pub fn set_ivf_flat_params(mut self, params: IvfFlatIndexParams) -> IndexParams {
        unsafe {
            (*self.inner).ivf_flat_params = params.0;
        }
        self.ivf_flat_params = Some(params);
        self
    }

    /// Supply IVF-PQ build parameters for the ANN tier.
    pub fn set_ivf_pq_params(mut self, params: IvfPqIndexParams) -> IndexParams {
        unsafe {
            (*self.inner).ivf_pq_params = params.0;
        }
        self.ivf_pq_params = Some(params);
        self
    }
}

impl fmt::Debug for IndexParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug trait here, default value will show the pointer address
        // for the inner params object which isn't that useful.
        write!(f, "IndexParams({:?})", unsafe { *self.inner })
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsTieredIndexParamsDestroy(self.inner) }) {
            write!(stderr(), "failed to call cuvsTieredIndexParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_params() {
        let params = IndexParams::new()
            .unwrap()
            .set_algo(AnnAlgo::CUVS_TIERED_INDEX_ALGO_CAGRA)
            .set_min_ann_rows(256)
            .set_create_ann_index_on_extend(true);

        unsafe {
            assert_eq!((*params.inner).algo, AnnAlgo::CUVS_TIERED_INDEX_ALGO_CAGRA);
            assert_eq!((*params.inner).min_ann_rows, 256);
            assert!((*params.inner).create_ann_index_on_extend);
        }
    }

    #[test]
    fn test_embedded_cagra_params_retained() {
        let cagra = CagraIndexParams::new().unwrap().set_graph_degree(32);
        let params = IndexParams::new()
            .unwrap()
            .set_algo(AnnAlgo::CUVS_TIERED_INDEX_ALGO_CAGRA)
            .set_cagra_params(cagra);

        // The embedded cagra params pointer must be live (not dangling) because
        // IndexParams retains ownership.
        unsafe {
            assert!(!(*params.inner).cagra_params.is_null());
            assert_eq!((*(*params.inner).cagra_params).graph_degree, 32);
        }
    }
}
