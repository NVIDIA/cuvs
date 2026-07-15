/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::distance_type::DistanceType;
use crate::error::{Result, check_cuvs};
use std::fmt;
use std::io::{Write, stderr};

pub struct Params(pub ffi::cuvsKMeansParams_t);

impl Params {
    /// Returns a new Params
    pub fn new() -> Result<Params> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsKMeansParams_t>::uninit();
            check_cuvs(ffi::cuvsKMeansParamsCreate(params.as_mut_ptr()))?;
            Ok(Params(params.assume_init()))
        }
    }

    /// DistanceType to use for fitting kmeans
    pub fn set_metric(self, metric: DistanceType) -> Params {
        unsafe {
            (*self.0).metric = metric;
        }
        self
    }

    /// The number of clusters to form as well as the number of centroids to generate (default:8).
    pub fn set_n_clusters(self, n_clusters: i32) -> Params {
        unsafe {
            (*self.0).n_clusters = n_clusters;
        }
        self
    }

    /// Maximum number of iterations of the k-means algorithm for a single run.
    pub fn set_max_iter(self, max_iter: i32) -> Params {
        unsafe {
            (*self.0).max_iter = max_iter;
        }
        self
    }

    /// Relative tolerance with regards to inertia to declare convergence.
    pub fn set_tol(self, tol: f64) -> Params {
        unsafe {
            (*self.0).tol = tol;
        }
        self
    }

    /// Number of instance k-means algorithm will be run with different seeds.
    pub fn set_n_init(self, n_init: i32) -> Params {
        unsafe {
            (*self.0).n_init = n_init;
        }
        self
    }

    /// Oversampling factor for use in the k-means|| algorithm
    pub fn set_oversampling_factor(self, oversampling_factor: f64) -> Params {
        unsafe {
            (*self.0).oversampling_factor = oversampling_factor;
        }
        self
    }

    /**
     * batch_samples and batch_centroids are used to tile 1NN computation which is
     * useful to optimize/control the memory footprint
     * Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0
     * then don't tile the centroids.
     */
    pub fn set_batch_samples(self, batch_samples: i32) -> Params {
        unsafe {
            (*self.0).batch_samples = batch_samples;
        }
        self
    }
    /// if 0 then batch_centroids = n_clusters
    pub fn set_batch_centroids(self, batch_centroids: i32) -> Params {
        unsafe {
            (*self.0).batch_centroids = batch_centroids;
        }
        self
    }

    /// Whether to use hierarchical (balanced) kmeans or not
    pub fn set_hierarchical(self, hierarchical: bool) -> Params {
        unsafe {
            (*self.0).hierarchical = hierarchical;
        }
        self
    }

    /// For hierarchical k-means , defines the number of training iterations
    pub fn set_hierarchical_n_iters(self, hierarchical_n_iters: i32) -> Params {
        unsafe {
            (*self.0).hierarchical_n_iters = hierarchical_n_iters;
        }
        self
    }
}

impl fmt::Debug for Params {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug trait here, default value will show the pointer address
        // for the inner params object which isn't that useful.
        write!(f, "Params({:?})", unsafe { *self.0 })
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsKMeansParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsKMeansParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params() {
        let params = Params::new().unwrap().set_n_clusters(128).set_hierarchical(true);

        unsafe {
            assert_eq!((*params.0).n_clusters, 128);
            assert_eq!((*params.0).hierarchical, true);
        }
    }
}
