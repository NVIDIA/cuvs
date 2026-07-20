/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter types for CAGRA index build and search.
//!
//! Each parameter type owns its C params handle directly. The generated `bon`
//! builder configures that handle in the constructor, so there is no duplicate
//! Rust field-bag to keep in sync with the FFI state. All setters are optional;
//! unset values retain the library defaults from the underlying C
//! `*ParamsCreate` functions. Out-of-range values are rejected by `build()` with
//! [`CagraError::Validation`].

use std::ffi::c_void;
use std::{fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;

use super::{CagraError, GraphBuildAlgo, HashMode, SearchAlgo};

// ---------------------------------------------------------------------------
// CompressionParams
// ---------------------------------------------------------------------------

/// VPQ (Vector-Product Quantization) compression parameters.
///
/// Attach to [`IndexParams`] to enable compressed dataset storage.
pub struct CompressionParams {
    handle: ffi::cuvsCagraCompressionParams_t,
}

#[bon]
impl CompressionParams {
    #[builder]
    pub fn new(
        pq_bits: Option<u32>,
        pq_dim: Option<u32>,
        vq_n_centers: Option<u32>,
        kmeans_n_iters: Option<u32>,
        vq_kmeans_trainset_fraction: Option<f64>,
        pq_kmeans_trainset_fraction: Option<f64>,
    ) -> Result<Self, CagraError> {
        if let Some(bits) = pq_bits
            && !(4..=16).contains(&bits)
        {
            return Err(CagraError::Validation(format!(
                "pq_bits must be within [4, 16], got {bits}"
            )));
        }

        let params = Self::try_new()?;
        unsafe {
            if let Some(v) = pq_bits {
                (*params.handle).pq_bits = v;
            }
            if let Some(v) = pq_dim {
                (*params.handle).pq_dim = v;
            }
            if let Some(v) = vq_n_centers {
                (*params.handle).vq_n_centers = v;
            }
            if let Some(v) = kmeans_n_iters {
                (*params.handle).kmeans_n_iters = v;
            }
            if let Some(v) = vq_kmeans_trainset_fraction {
                (*params.handle).vq_kmeans_trainset_fraction = v;
            }
            if let Some(v) = pq_kmeans_trainset_fraction {
                (*params.handle).pq_kmeans_trainset_fraction = v;
            }
        }

        Ok(params)
    }
}

impl CompressionParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraCompressionParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    fn handle(&self) -> ffi::cuvsCagraCompressionParams_t {
        self.handle
    }
}

impl fmt::Debug for CompressionParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CompressionParams").field(unsafe { &*self.handle }).finish()
    }
}

impl Drop for CompressionParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsCagraCompressionParamsDestroy(self.handle) };
    }
}

#[derive(Debug)]
enum RequestedGraphBuild {
    Auto,
    NnDescent { iterations: Option<usize> },
    IterativeCagraSearch,
    Ace,
    IvfPq,
}

// ---------------------------------------------------------------------------
// IndexParams
// ---------------------------------------------------------------------------

/// Parameters for building a CAGRA index.
///
/// ```ignore
/// use cuvs::neighbors::cagra::IndexParams;
/// use cuvs::distance::DistanceType;
///
/// let params = IndexParams::builder()
///     .metric(DistanceType::InnerProduct)
///     .graph_degree(64)
///     .build()?;
/// ```
pub struct IndexParams {
    handle: ffi::cuvsCagraIndexParams_t,
    // Saved so Drop can restore the C-owned default IVF-PQ params before the C
    // destructor runs; other graph strategies temporarily replace it with null.
    default_graph_build_params: *mut c_void,
    compression: Option<CompressionParams>,
}

#[bon]
impl IndexParams {
    #[builder]
    pub fn new(
        metric: Option<DistanceType>,
        intermediate_graph_degree: Option<usize>,
        graph_degree: Option<usize>,
        compression: Option<CompressionParams>,
        #[builder(setters(vis = "", some_fn = graph_build_internal))] graph_build: Option<
            RequestedGraphBuild,
        >,
    ) -> Result<Self, CagraError> {
        if let Some(d) = graph_degree
            && d == 0
        {
            return Err(CagraError::Validation("graph_degree must be > 0".into()));
        }

        if let (Some(inter), Some(graph)) = (intermediate_graph_degree, graph_degree)
            && inter < graph
        {
            return Err(CagraError::Validation(format!(
                "intermediate_graph_degree ({inter}) must be >= graph_degree ({graph})"
            )));
        }

        if let Some(RequestedGraphBuild::NnDescent { iterations: Some(0) }) = &graph_build {
            return Err(CagraError::Validation("nn_descent_niter must be > 0".into()));
        }

        let metric_supports_compression = metric.is_none_or(|v| v == DistanceType::L2Expanded);
        if compression.is_some() && !metric_supports_compression {
            return Err(CagraError::Validation(
                "VPQ compression is only supported with L2Expanded distance metric".into(),
            ));
        }

        let mut params = Self::try_new()?;

        unsafe {
            if let Some(v) = metric {
                (*params.handle).metric = v.into();
            }
            if let Some(v) = intermediate_graph_degree {
                (*params.handle).intermediate_graph_degree = v;
            }
            if let Some(v) = graph_degree {
                (*params.handle).graph_degree = v;
            }
        }

        if let Some(compression) = compression {
            unsafe { (*params.handle).compression = compression.handle() };
            params.compression = Some(compression);
        }

        params.apply_graph_build(graph_build)?;
        Ok(params)
    }
}

use index_params_builder::{IsUnset, SetGraphBuild, State};

impl<S: State> IndexParamsBuilder<S> {
    pub fn auto(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::Auto)
    }

    pub fn nn_descent(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::NnDescent { iterations: None })
    }

    pub fn nn_descent_with(self, iterations: usize) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::NnDescent { iterations: Some(iterations) })
    }

    pub fn iterative_cagra_search(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::IterativeCagraSearch)
    }

    pub fn ace(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::Ace)
    }

    pub fn ivf_pq(self) -> IndexParamsBuilder<SetGraphBuild<S>>
    where
        S::GraphBuild: IsUnset,
    {
        self.graph_build_internal(RequestedGraphBuild::IvfPq)
    }
}

impl IndexParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraIndexParamsCreate(&mut handle) })?;
        let default_graph_build_params = unsafe { (*handle).graph_build_params };
        Ok(Self { handle, default_graph_build_params, compression: None })
    }

    pub(super) fn handle(&self) -> ffi::cuvsCagraIndexParams_t {
        self.handle
    }

    fn apply_graph_build(
        &mut self,
        graph_build: Option<RequestedGraphBuild>,
    ) -> Result<(), CagraError> {
        let Some(graph_build) = graph_build else {
            return Ok(());
        };
        match graph_build {
            RequestedGraphBuild::Auto => unsafe {
                (*self.handle).build_algo = GraphBuildAlgo::Auto.into();
                (*self.handle).graph_build_params = ptr::null_mut();
            },
            RequestedGraphBuild::NnDescent { iterations } => unsafe {
                (*self.handle).build_algo = GraphBuildAlgo::NnDescent.into();
                (*self.handle).graph_build_params = ptr::null_mut();
                if let Some(value) = iterations {
                    (*self.handle).nn_descent_niter = value;
                }
            },
            RequestedGraphBuild::IterativeCagraSearch => unsafe {
                (*self.handle).build_algo = GraphBuildAlgo::IterativeCagraSearch.into();
                (*self.handle).graph_build_params = ptr::null_mut();
            },
            RequestedGraphBuild::Ace => unsafe {
                (*self.handle).build_algo = GraphBuildAlgo::Ace.into();
                (*self.handle).graph_build_params = ptr::null_mut();
            },
            RequestedGraphBuild::IvfPq => unsafe {
                (*self.handle).build_algo = GraphBuildAlgo::IvfPq.into();
                (*self.handle).graph_build_params = self.default_graph_build_params;
            },
        }
        Ok(())
    }
}

impl fmt::Debug for IndexParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IndexParams").field(unsafe { &*self.handle }).finish()
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        unsafe {
            (*self.handle).graph_build_params = self.default_graph_build_params;
            (*self.handle).build_algo = ffi::cuvsCagraGraphBuildAlgo::IVF_PQ;
            let _ = ffi::cuvsCagraIndexParamsDestroy(self.handle);
        }
    }
}

// ---------------------------------------------------------------------------
// SearchParams
// ---------------------------------------------------------------------------

/// Parameters for searching a CAGRA index.
///
/// ```ignore
/// use cuvs::neighbors::cagra::SearchParams;
///
/// let params = SearchParams::builder().itopk_size(128).build()?;
/// ```
pub struct SearchParams {
    handle: ffi::cuvsCagraSearchParams_t,
}

#[bon]
impl SearchParams {
    #[builder]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_queries: Option<usize>,
        itopk_size: Option<usize>,
        max_iterations: Option<usize>,
        algo: Option<SearchAlgo>,
        team_size: Option<usize>,
        min_iterations: Option<usize>,
        thread_block_size: Option<usize>,
        hashmap_mode: Option<HashMode>,
        hashmap_min_bitlen: Option<usize>,
        hashmap_max_fill_rate: Option<f32>,
        num_random_samplings: Option<u32>,
        rand_xor_mask: Option<u64>,
    ) -> Result<Self, CagraError> {
        let params = Self::try_new()?;

        let effective_algo = algo.unwrap_or_else(|| unsafe { (*params.handle).algo.into() });
        let effective_hashmap_mode =
            hashmap_mode.unwrap_or_else(|| unsafe { (*params.handle).hashmap_mode.into() });

        if let Some(n) = itopk_size
            && effective_algo == SearchAlgo::SingleCta
            && n > 512
        {
            return Err(CagraError::Validation(format!(
                "itopk_size cannot be larger than 512 for SingleCta, got {n}"
            )));
        }

        if let Some(n) = team_size
            && !matches!(n, 0 | 8 | 16 | 32)
        {
            return Err(CagraError::Validation(format!(
                "team_size must be 0 (auto), 8, 16, or 32, got {n}"
            )));
        }

        if let Some(n) = thread_block_size
            && !matches!(n, 0 | 64 | 128 | 256 | 512 | 1024)
        {
            return Err(CagraError::Validation(format!(
                "thread_block_size must be 0, 64, 128, 256, 512, or 1024, got {n}"
            )));
        }

        if let Some(bitlen) = hashmap_min_bitlen
            && bitlen > 20
        {
            return Err(CagraError::Validation(format!(
                "hashmap_min_bitlen must be <= 20, got {bitlen}"
            )));
        }

        if let Some(rate) = hashmap_max_fill_rate
            && !(0.1..0.9).contains(&rate)
        {
            return Err(CagraError::Validation(format!(
                "hashmap_max_fill_rate must be in [0.1, 0.9), got {rate}"
            )));
        }

        if effective_algo == SearchAlgo::MultiCta && effective_hashmap_mode == HashMode::Small {
            return Err(CagraError::Validation(
                "`small_hash` is not available when 'search_mode' is \"multi-cta\"".into(),
            ));
        }

        unsafe {
            if let Some(v) = max_queries {
                (*params.handle).max_queries = v;
            }
            if let Some(v) = itopk_size {
                (*params.handle).itopk_size = v;
            }
            if let Some(v) = max_iterations {
                (*params.handle).max_iterations = v;
            }
            if let Some(v) = algo {
                (*params.handle).algo = v.into();
            }
            if let Some(v) = team_size {
                (*params.handle).team_size = v;
            }
            if let Some(v) = min_iterations {
                (*params.handle).min_iterations = v;
            }
            if let Some(v) = thread_block_size {
                (*params.handle).thread_block_size = v;
            }
            if let Some(v) = hashmap_mode {
                (*params.handle).hashmap_mode = v.into();
            }
            if let Some(v) = hashmap_min_bitlen {
                (*params.handle).hashmap_min_bitlen = v;
            }
            if let Some(v) = hashmap_max_fill_rate {
                (*params.handle).hashmap_max_fill_rate = v;
            }
            if let Some(v) = num_random_samplings {
                (*params.handle).num_random_samplings = v;
            }
            if let Some(v) = rand_xor_mask {
                (*params.handle).rand_xor_mask = v;
            }
        }

        Ok(params)
    }
}

impl SearchParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, CagraError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsCagraSearchParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsCagraSearchParams_t {
        self.handle
    }
}

impl fmt::Debug for SearchParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SearchParams").field(unsafe { &*self.handle }).finish()
    }
}

impl Drop for SearchParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsCagraSearchParamsDestroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_params_all_defaults() {
        let params = IndexParams::try_new().unwrap();
        unsafe {
            assert_eq!((*params.handle).metric, ffi::cuvsDistanceType::L2Expanded);
            assert_eq!((*params.handle).graph_degree, 64);
        }
    }

    #[test]
    fn index_params_with_values() {
        let params = IndexParams::builder()
            .metric(DistanceType::InnerProduct)
            .graph_degree(64)
            .intermediate_graph_degree(128)
            .nn_descent_with(10)
            .build()
            .unwrap();

        unsafe {
            assert_eq!((*params.handle).metric, ffi::cuvsDistanceType::InnerProduct);
            assert_eq!((*params.handle).graph_degree, 64);
            assert_eq!((*params.handle).intermediate_graph_degree, 128);
            assert_eq!((*params.handle).build_algo, ffi::cuvsCagraGraphBuildAlgo::NN_DESCENT);
            assert_eq!((*params.handle).nn_descent_niter, 10);
        }
    }

    #[test]
    fn index_params_rejects_zero_graph_degree() {
        let err = IndexParams::builder().graph_degree(0).build().unwrap_err();
        assert!(err.to_string().contains("graph_degree must be > 0"));
    }

    #[test]
    fn index_params_rejects_invalid_intermediate_degree() {
        let err = IndexParams::builder()
            .graph_degree(64)
            .intermediate_graph_degree(32)
            .build()
            .unwrap_err();
        assert!(
            err.to_string().contains("intermediate_graph_degree (32) must be >= graph_degree (64)")
        );
    }

    #[test]
    fn index_params_rejects_zero_niter() {
        let err = IndexParams::builder().nn_descent_with(0).build().unwrap_err();
        assert!(err.to_string().contains("nn_descent_niter must be > 0"));
    }

    #[test]
    fn index_params_rejects_non_l2_metric_with_compression() {
        let compression = CompressionParams::builder().pq_bits(8).build().unwrap();
        let err = IndexParams::builder()
            .metric(DistanceType::InnerProduct)
            .compression(compression)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("VPQ compression is only supported with L2Expanded"));
    }

    #[test]
    fn index_params_with_compression() {
        let params = IndexParams::builder()
            .compression(CompressionParams::builder().pq_bits(4).pq_dim(8).build().unwrap())
            .build()
            .unwrap();
        unsafe {
            let c = (*params.handle).compression;
            assert!(!c.is_null());
            assert_eq!((*c).pq_bits, 4);
            assert_eq!((*c).pq_dim, 8);
        }
    }

    #[test]
    fn index_params_selects_default_graph_build_strategies() {
        let ace_params = IndexParams::builder().ace().build().unwrap();
        unsafe {
            assert_eq!((*ace_params.handle).build_algo, ffi::cuvsCagraGraphBuildAlgo::ACE);
            assert!((*ace_params.handle).graph_build_params.is_null());
        }

        let ivf_pq_params = IndexParams::builder().ivf_pq().build().unwrap();
        unsafe {
            assert_eq!((*ivf_pq_params.handle).build_algo, ffi::cuvsCagraGraphBuildAlgo::IVF_PQ);
            assert!(!(*ivf_pq_params.handle).graph_build_params.is_null());
        }
    }

    #[test]
    fn compression_params_rejects_pq_bits_below_range() {
        let err = CompressionParams::builder().pq_bits(3).build().unwrap_err();
        assert!(err.to_string().contains("pq_bits"));
    }

    #[test]
    fn search_params_all_defaults() {
        let params = SearchParams::try_new().unwrap();
        unsafe {
            assert_eq!((*params.handle).itopk_size, 64);
            assert_eq!((*params.handle).algo, ffi::cuvsCagraSearchAlgo::SINGLE_CTA);
        }
    }

    #[test]
    fn search_params_rejects_invalid_team_size() {
        let err = SearchParams::builder().team_size(4).build().unwrap_err();
        assert!(err.to_string().contains("team_size must be"));
    }

    #[test]
    fn search_params_rejects_single_cta_itopk_above_limit() {
        let err = SearchParams::builder()
            .algo(SearchAlgo::SingleCta)
            .itopk_size(513)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn search_params_rejects_small_hash_with_multi_cta() {
        let err = SearchParams::builder()
            .algo(SearchAlgo::MultiCta)
            .hashmap_mode(HashMode::Small)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("small_hash"));
    }
}
