/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter types for IVF-PQ build and search.
//!
//! All setters are optional; unset values retain the library defaults from the
//! underlying C `*ParamsCreate` functions.

use std::{fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;

use super::{CodebookGen, InternalDistanceDType, IvfPqError, ListLayout, LutDType};

/// Parameters for building an IVF-PQ index.
pub struct IndexParams {
    handle: ffi::cuvsIvfPqIndexParams_t,
}

#[bon]
impl IndexParams {
    #[builder]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_lists: Option<u32>,
        metric: Option<DistanceType>,
        kmeans_n_iters: Option<u32>,
        kmeans_trainset_fraction: Option<f64>,
        pq_bits: Option<u32>,
        pq_dim: Option<u32>,
        codebook_kind: Option<CodebookGen>,
        codes_layout: Option<ListLayout>,
        force_random_rotation: Option<bool>,
        max_train_points_per_pq_code: Option<u32>,
        add_data_on_build: Option<bool>,
    ) -> Result<Self, IvfPqError> {
        let params = Self::try_new()?;
        unsafe {
            if let Some(v) = n_lists {
                (*params.handle).n_lists = v;
            }
            if let Some(v) = metric {
                (*params.handle).metric = v.into();
                (*params.handle).metric_arg = v.metric_arg();
            }
            if let Some(v) = kmeans_n_iters {
                (*params.handle).kmeans_n_iters = v;
            }
            if let Some(v) = kmeans_trainset_fraction {
                (*params.handle).kmeans_trainset_fraction = v;
            }
            if let Some(v) = pq_bits {
                (*params.handle).pq_bits = v;
            }
            if let Some(v) = pq_dim {
                (*params.handle).pq_dim = v;
            }
            if let Some(v) = codebook_kind {
                (*params.handle).codebook_kind = v.into();
            }
            if let Some(v) = codes_layout {
                (*params.handle).codes_layout = v.into();
            }
            if let Some(v) = force_random_rotation {
                (*params.handle).force_random_rotation = v;
            }
            if let Some(v) = max_train_points_per_pq_code {
                (*params.handle).max_train_points_per_pq_code = v;
            }
            if let Some(v) = add_data_on_build {
                (*params.handle).add_data_on_build = v;
            }
        }
        Ok(params)
    }
}

impl IndexParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, IvfPqError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfPqIndexParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsIvfPqIndexParams_t {
        self.handle
    }
}

impl fmt::Debug for IndexParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IndexParams").field(unsafe { &*self.handle }).finish()
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsIvfPqIndexParamsDestroy(self.handle) };
    }
}

/// Parameters for searching an IVF-PQ index.
pub struct SearchParams {
    handle: ffi::cuvsIvfPqSearchParams_t,
}

#[bon]
impl SearchParams {
    #[builder]
    pub fn new(
        n_probes: Option<u32>,
        lut_dtype: Option<LutDType>,
        internal_distance_dtype: Option<InternalDistanceDType>,
    ) -> Result<Self, IvfPqError> {
        let params = Self::try_new()?;
        unsafe {
            if let Some(v) = n_probes {
                (*params.handle).n_probes = v;
            }
            if let Some(v) = lut_dtype {
                (*params.handle).lut_dtype = v.into();
            }
            if let Some(v) = internal_distance_dtype {
                (*params.handle).internal_distance_dtype = v.into();
            }
        }
        Ok(params)
    }
}

impl SearchParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, IvfPqError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfPqSearchParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsIvfPqSearchParams_t {
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
        let _ = unsafe { ffi::cuvsIvfPqSearchParamsDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_params_with_values() {
        let params = IndexParams::builder().n_lists(128).add_data_on_build(false).build().unwrap();
        unsafe {
            assert_eq!((*params.handle).n_lists, 128);
            assert!(!(*params.handle).add_data_on_build);
        }
    }

    #[test]
    fn search_params_with_values() {
        let params = SearchParams::builder().n_probes(128).build().unwrap();
        unsafe {
            assert_eq!((*params.handle).n_probes, 128);
        }
    }
}
