/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::distance_type::DistanceType;
use crate::error::{Result, check_cuvs};
use std::fmt;
use std::io::{Write, stderr};

pub struct IndexParams(pub ffi::cuvsIvfSqIndexParams_t);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsIvfSqIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsIvfSqIndexParamsCreate(params.as_mut_ptr()))?;
            Ok(IndexParams(params.assume_init()))
        }
    }

    /// DistanceType to use for building the index
    pub fn set_metric(self, metric: DistanceType) -> IndexParams {
        unsafe {
            (*self.0).metric = metric;
        }
        self
    }

    /// The argument used by some distance metrics
    pub fn set_metric_arg(self, metric_arg: f32) -> IndexParams {
        unsafe {
            (*self.0).metric_arg = metric_arg;
        }
        self
    }

    /// The number of clusters used in the coarse quantizer.
    pub fn set_n_lists(self, n_lists: u32) -> IndexParams {
        unsafe {
            (*self.0).n_lists = n_lists;
        }
        self
    }

    /// The number of iterations searching for kmeans centers during index building.
    pub fn set_kmeans_n_iters(self, kmeans_n_iters: u32) -> IndexParams {
        unsafe {
            (*self.0).kmeans_n_iters = kmeans_n_iters;
        }
        self
    }

    /// The number of data vectors per cluster to use during iterative kmeans
    /// building. The index uses at most `n_lists * max_train_points_per_cluster`
    /// rows for training.
    pub fn set_max_train_points_per_cluster(
        self,
        max_train_points_per_cluster: u32,
    ) -> IndexParams {
        unsafe {
            (*self.0).max_train_points_per_cluster = max_train_points_per_cluster;
        }
        self
    }

    /// After training the quantizer and clustering, we will populate
    /// the index with the dataset if add_data_on_build == true, otherwise
    /// the index is left empty, and the extend method can be used
    /// to add new vectors to the index.
    pub fn set_add_data_on_build(self, add_data_on_build: bool) -> IndexParams {
        unsafe {
            (*self.0).add_data_on_build = add_data_on_build;
        }
        self
    }

    /// If set to `true`, the algorithm always allocates the minimum amount of
    /// memory required to store the given number of records, trading reduced GPU
    /// memory usage for more frequent reallocations during `extend`.
    pub fn set_conservative_memory_allocation(
        self,
        conservative_memory_allocation: bool,
    ) -> IndexParams {
        unsafe {
            (*self.0).conservative_memory_allocation = conservative_memory_allocation;
        }
        self
    }
}

impl fmt::Debug for IndexParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug trait here, default value will show the pointer address
        // for the inner params object which isn't that useful.
        write!(f, "IndexParams({:?})", unsafe { *self.0 })
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfSqIndexParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsIvfSqIndexParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_params() {
        let params = IndexParams::new().unwrap().set_n_lists(128).set_add_data_on_build(false);

        unsafe {
            assert_eq!((*params.0).n_lists, 128);
            assert_eq!((*params.0).add_data_on_build, false);
        }
    }
}
