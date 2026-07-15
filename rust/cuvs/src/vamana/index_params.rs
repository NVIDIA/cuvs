/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::distance_type::DistanceType;
use crate::error::{Result, check_cuvs};
use std::fmt;
use std::io::{Write, stderr};

pub struct IndexParams(pub ffi::cuvsVamanaIndexParams_t);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsVamanaIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsVamanaIndexParamsCreate(params.as_mut_ptr()))?;
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

    /// Maximum degree of output graph corresponds to the R parameter in the original Vamana
    /// literature.
    pub fn set_graph_degree(self, graph_degree: u32) -> IndexParams {
        unsafe {
            (*self.0).graph_degree = graph_degree;
        }
        self
    }

    /// Maximum number of visited nodes per search corresponds to the L parameter in the Vamana
    /// literature
    pub fn set_visited_size(self, visited_size: u32) -> IndexParams {
        unsafe {
            (*self.0).visited_size = visited_size;
        }
        self
    }

    /// Number of Vamana vector insertion iterations (each iteration inserts all vectors).
    pub fn set_vamana_iters(self, vamana_iters: f32) -> IndexParams {
        unsafe {
            (*self.0).vamana_iters = vamana_iters;
        }
        self
    }

    /// Alpha for pruning parameter
    pub fn set_alpha(self, alpha: f32) -> IndexParams {
        unsafe {
            (*self.0).alpha = alpha;
        }
        self
    }

    /// Maximum fraction of dataset inserted per batch.
    /// Larger max batch decreases graph quality, but improves speed
    pub fn set_max_fraction(self, max_fraction: f32) -> IndexParams {
        unsafe {
            (*self.0).max_fraction = max_fraction;
        }
        self
    }

    /// Base of growth rate of batch sizes
    pub fn set_batch_base(self, batch_base: f32) -> IndexParams {
        unsafe {
            (*self.0).batch_base = batch_base;
        }
        self
    }

    /// Size of candidate queue structure - should be (2^x)-1
    pub fn set_queue_size(self, queue_size: u32) -> IndexParams {
        unsafe {
            (*self.0).queue_size = queue_size;
        }
        self
    }

    /// Max batchsize of reverse edge processing (reduces memory footprint)
    pub fn set_reverse_batchsize(self, reverse_batchsize: u32) -> IndexParams {
        unsafe {
            (*self.0).reverse_batchsize = reverse_batchsize;
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsVamanaIndexParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsVamanaIndexParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_params() {
        let params = IndexParams::new().unwrap().set_alpha(1.0).set_visited_size(128);

        unsafe {
            assert_eq!((*params.0).alpha, 1.0);
            assert_eq!((*params.0).visited_size, 128);
        }
    }
}
