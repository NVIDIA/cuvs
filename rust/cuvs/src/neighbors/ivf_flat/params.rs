/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter types for IVF-Flat build and search.
//!
//! All setters are optional; unset values retain the library defaults from the
//! underlying C `*ParamsCreate` functions.

use std::{fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;

use super::IvfFlatError;

/// Parameters for building an IVF-Flat index.
pub struct IndexParams {
    handle: ffi::cuvsIvfFlatIndexParams_t,
}

#[bon]
impl IndexParams {
    #[builder]
    pub fn new(
        n_lists: Option<u32>,
        metric: Option<DistanceType>,
        kmeans_n_iters: Option<u32>,
        kmeans_trainset_fraction: Option<f64>,
        add_data_on_build: Option<bool>,
    ) -> Result<Self, IvfFlatError> {
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
            if let Some(v) = add_data_on_build {
                (*params.handle).add_data_on_build = v;
            }
        }
        Ok(params)
    }
}

impl IndexParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, IvfFlatError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfFlatIndexParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsIvfFlatIndexParams_t {
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
        let _ = unsafe { ffi::cuvsIvfFlatIndexParamsDestroy(self.handle) };
    }
}

/// Parameters for searching an IVF-Flat index.
pub struct SearchParams {
    handle: ffi::cuvsIvfFlatSearchParams_t,
}

#[bon]
impl SearchParams {
    #[builder]
    pub fn new(n_probes: Option<u32>) -> Result<Self, IvfFlatError> {
        let params = Self::try_new()?;
        unsafe {
            if let Some(v) = n_probes {
                (*params.handle).n_probes = v;
            }
        }
        Ok(params)
    }
}

impl SearchParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, IvfFlatError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsIvfFlatSearchParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsIvfFlatSearchParams_t {
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
        let _ = unsafe { ffi::cuvsIvfFlatSearchParamsDestroy(self.handle) };
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
