/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builder-pattern parameter type for k-means.
//!
//! All setters are optional; unset values retain the library defaults from the
//! underlying C `cuvsKMeansParamsCreate`.

use std::{fmt, ptr};

use bon::bon;

use crate::distance::DistanceType;
use crate::error::check_cuvs;

use super::KMeansError;

/// Parameters for k-means fitting and prediction.
pub struct Params {
    handle: ffi::cuvsKMeansParams_t,
}

#[bon]
impl Params {
    #[builder]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metric: Option<DistanceType>,
        n_clusters: Option<i32>,
        max_iter: Option<i32>,
        tol: Option<f64>,
        n_init: Option<i32>,
        oversampling_factor: Option<f64>,
        batch_samples: Option<i32>,
        batch_centroids: Option<i32>,
        hierarchical: Option<bool>,
        hierarchical_n_iters: Option<i32>,
    ) -> Result<Self, KMeansError> {
        if let Some(n) = n_clusters
            && n <= 0
        {
            return Err(KMeansError::Validation("n_clusters must be > 0".into()));
        }
        if let Some(n) = max_iter
            && n < 0
        {
            return Err(KMeansError::Validation("max_iter must be >= 0".into()));
        }
        if let Some(n) = n_init
            && n <= 0
        {
            return Err(KMeansError::Validation("n_init must be > 0".into()));
        }
        if let Some(n) = hierarchical_n_iters
            && n < 0
        {
            return Err(KMeansError::Validation("hierarchical_n_iters must be >= 0".into()));
        }

        let params = Self::try_new()?;
        unsafe {
            if let Some(v) = metric {
                (*params.handle).metric = v.into();
            }
            if let Some(v) = n_clusters {
                (*params.handle).n_clusters = v;
            }
            if let Some(v) = max_iter {
                (*params.handle).max_iter = v;
            }
            if let Some(v) = tol {
                (*params.handle).tol = v;
            }
            if let Some(v) = n_init {
                (*params.handle).n_init = v;
            }
            if let Some(v) = oversampling_factor {
                (*params.handle).oversampling_factor = v;
            }
            if let Some(v) = batch_samples {
                (*params.handle).batch_samples = v;
            }
            if let Some(v) = batch_centroids {
                (*params.handle).batch_centroids = v;
            }
            if let Some(v) = hierarchical {
                (*params.handle).hierarchical = v;
            }
            if let Some(v) = hierarchical_n_iters {
                (*params.handle).hierarchical_n_iters = v;
            }
        }
        Ok(params)
    }
}

impl Params {
    /// Allocate parameters populated with the library defaults.
    pub fn try_new() -> Result<Self, KMeansError> {
        let mut handle = ptr::null_mut();
        check_cuvs(unsafe { ffi::cuvsKMeansParamsCreate(&mut handle) })?;
        Ok(Self { handle })
    }

    pub(super) fn handle(&self) -> ffi::cuvsKMeansParams_t {
        self.handle
    }
}

impl fmt::Debug for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Params").field(unsafe { &*self.handle }).finish()
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsKMeansParamsDestroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_with_values() {
        let params = Params::builder().n_clusters(128).hierarchical(true).build().unwrap();
        unsafe {
            assert_eq!((*params.handle).n_clusters, 128);
            assert!((*params.handle).hierarchical);
        }
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(matches!(Params::builder().n_clusters(0).build(), Err(KMeansError::Validation(_))));
        assert!(matches!(Params::builder().max_iter(-1).build(), Err(KMeansError::Validation(_))));
        assert!(matches!(Params::builder().n_init(0).build(), Err(KMeansError::Validation(_))));
        assert!(matches!(
            Params::builder().hierarchical_n_iters(-1).build(),
            Err(KMeansError::Validation(_))
        ));
    }
}
