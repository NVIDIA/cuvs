/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter type for Vamana index build.
//!
//! All setters are optional; unset values retain the library defaults from the
//! underlying C `cuvsVamanaIndexParamsCreate`.

use std::{fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;

use super::VamanaError;

/// Parameters for building a Vamana index.
pub struct IndexParams {
    handle: ffi::cuvsVamanaIndexParams_t,
}

#[bon]
impl IndexParams {
    #[builder]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metric: Option<DistanceType>,
        graph_degree: Option<u32>,
        visited_size: Option<u32>,
        vamana_iters: Option<f32>,
        alpha: Option<f32>,
        max_fraction: Option<f32>,
        batch_base: Option<f32>,
        queue_size: Option<u32>,
        reverse_batchsize: Option<u32>,
    ) -> Result<Self, VamanaError> {
        let params = Self::try_new()?;
        unsafe {
            if let Some(v) = metric {
                (*params.handle).metric = v.into();
            }
            if let Some(v) = graph_degree {
                (*params.handle).graph_degree = v;
            }
            if let Some(v) = visited_size {
                (*params.handle).visited_size = v;
            }
            if let Some(v) = vamana_iters {
                (*params.handle).vamana_iters = v;
            }
            if let Some(v) = alpha {
                (*params.handle).alpha = v;
            }
            if let Some(v) = max_fraction {
                (*params.handle).max_fraction = v;
            }
            if let Some(v) = batch_base {
                (*params.handle).batch_base = v;
            }
            if let Some(v) = queue_size {
                (*params.handle).queue_size = v;
            }
            if let Some(v) = reverse_batchsize {
                (*params.handle).reverse_batchsize = v;
            }
        }
        Ok(params)
    }
}

impl IndexParams {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, VamanaError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsVamanaIndexParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsVamanaIndexParams_t {
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
        let _ = unsafe { ffi::cuvsVamanaIndexParamsDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_params_with_values() {
        let params = IndexParams::builder().alpha(1.0).visited_size(128).build().unwrap();
        unsafe {
            assert_eq!((*params.handle).alpha, 1.0);
            assert_eq!((*params.handle).visited_size, 128);
        }
    }
}
