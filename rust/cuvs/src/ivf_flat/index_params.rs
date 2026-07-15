/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::distance_type::DistanceType;
use crate::error::{Result, check_cuvs};
use std::fmt;
use std::io::{Write, stderr};

pub struct IndexParams(pub ffi::cuvsIvfFlatIndexParams_t);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsIvfFlatIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsIvfFlatIndexParamsCreate(params.as_mut_ptr()))?;
            Ok(IndexParams(params.assume_init()))
        }
    }

    /// The number of clusters used in the coarse quantizer.
    pub fn set_n_lists(self, n_lists: u32) -> IndexParams {
        unsafe {
            (*self.0).n_lists = n_lists;
        }
        self
    }

    /// DistanceType to use for building the index
    pub fn set_metric(self, metric: DistanceType) -> IndexParams {
        unsafe {
            (*self.0).metric = metric;
        }
        self
    }

    /// The number of iterations searching for kmeans centers during index building.
    pub fn set_metric_arg(self, metric_arg: f32) -> IndexParams {
        unsafe {
            (*self.0).metric_arg = metric_arg;
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

    /// If kmeans_trainset_fraction is less than 1, then the dataset is
    /// subsampled, and only n_samples * kmeans_trainset_fraction rows
    /// are used for training.
    pub fn set_kmeans_trainset_fraction(self, kmeans_trainset_fraction: f64) -> IndexParams {
        unsafe {
            (*self.0).kmeans_trainset_fraction = kmeans_trainset_fraction;
        }
        self
    }

    /// After training the coarse and fine quantizers, we will populate
    /// the index with the dataset if add_data_on_build == true, otherwise
    /// the index is left empty, and the extend method can be used
    /// to add new vectors to the index.
    pub fn set_add_data_on_build(self, add_data_on_build: bool) -> IndexParams {
        unsafe {
            (*self.0).add_data_on_build = add_data_on_build;
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfFlatIndexParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsIvfFlatIndexParamsDestroy {:?}", e)
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
