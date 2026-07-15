/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::{Result, check_cuvs};
use std::fmt;
use std::io::{Write, stderr};

pub type BuildAlgo = ffi::cuvsCagraGraphBuildAlgo;

/// Supplemental parameters to build CAGRA Index
pub struct CompressionParams(pub ffi::cuvsCagraCompressionParams_t);

impl CompressionParams {
    /// Returns a new CompressionParams
    pub fn new() -> Result<CompressionParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsCagraCompressionParams_t>::uninit();
            check_cuvs(ffi::cuvsCagraCompressionParamsCreate(params.as_mut_ptr()))?;
            Ok(CompressionParams(params.assume_init()))
        }
    }

    /// The bit length of the vector element after compression by PQ.
    pub fn set_pq_bits(self, pq_bits: u32) -> CompressionParams {
        unsafe {
            (*self.0).pq_bits = pq_bits;
        }
        self
    }

    /// The dimensionality of the vector after compression by PQ. When zero,
    /// an optimal value is selected using a heuristic.
    pub fn set_pq_dim(self, pq_dim: u32) -> CompressionParams {
        unsafe {
            (*self.0).pq_dim = pq_dim;
        }
        self
    }

    /// Vector Quantization (VQ) codebook size - number of "coarse cluster
    /// centers". When zero, an optimal value is selected using a heuristic.
    pub fn set_vq_n_centers(self, vq_n_centers: u32) -> CompressionParams {
        unsafe {
            (*self.0).vq_n_centers = vq_n_centers;
        }
        self
    }

    /// The number of iterations searching for kmeans centers (both VQ & PQ
    /// phases).
    pub fn set_kmeans_n_iters(self, kmeans_n_iters: u32) -> CompressionParams {
        unsafe {
            (*self.0).kmeans_n_iters = kmeans_n_iters;
        }
        self
    }

    /// The fraction of data to use during iterative kmeans building (VQ
    /// phase). When zero, an optimal value is selected using a heuristic.
    pub fn set_vq_kmeans_trainset_fraction(
        self,
        vq_kmeans_trainset_fraction: f64,
    ) -> CompressionParams {
        unsafe {
            (*self.0).vq_kmeans_trainset_fraction = vq_kmeans_trainset_fraction;
        }
        self
    }

    /// The fraction of data to use during iterative kmeans building (PQ
    /// phase). When zero, an optimal value is selected using a heuristic.
    pub fn set_pq_kmeans_trainset_fraction(
        self,
        pq_kmeans_trainset_fraction: f64,
    ) -> CompressionParams {
        unsafe {
            (*self.0).pq_kmeans_trainset_fraction = pq_kmeans_trainset_fraction;
        }
        self
    }
}

pub struct IndexParams(pub ffi::cuvsCagraIndexParams_t, Option<CompressionParams>);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsCagraIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexParamsCreate(params.as_mut_ptr()))?;
            Ok(IndexParams(params.assume_init(), None))
        }
    }

    /// Degree of input graph for pruning
    pub fn set_intermediate_graph_degree(self, intermediate_graph_degree: usize) -> IndexParams {
        unsafe {
            (*self.0).intermediate_graph_degree = intermediate_graph_degree;
        }
        self
    }

    /// Degree of output graph
    pub fn set_graph_degree(self, graph_degree: usize) -> IndexParams {
        unsafe {
            (*self.0).graph_degree = graph_degree;
        }
        self
    }

    /// ANN algorithm to build knn graph
    pub fn set_build_algo(self, build_algo: BuildAlgo) -> IndexParams {
        unsafe {
            (*self.0).build_algo = build_algo;
        }
        self
    }

    /// Number of iterations to run if building with NN_DESCENT
    pub fn set_nn_descent_niter(self, nn_descent_niter: usize) -> IndexParams {
        unsafe {
            (*self.0).nn_descent_niter = nn_descent_niter;
        }
        self
    }

    pub fn set_compression(mut self, compression: CompressionParams) -> IndexParams {
        unsafe {
            (*self.0).compression = compression.0;
        }
        // Note: we're moving the ownership of compression here to avoid having it cleaned up
        // and leaving a dangling pointer
        self.1 = Some(compression);
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

impl fmt::Debug for CompressionParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CompressionParams({:?})", unsafe { *self.0 })
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsCagraIndexParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

impl Drop for CompressionParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraCompressionParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsCagraCompressionParamsDestroy {:?}", e)
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
            .set_intermediate_graph_degree(128)
            .set_graph_degree(16)
            .set_build_algo(BuildAlgo::NN_DESCENT)
            .set_nn_descent_niter(10)
            .set_compression(CompressionParams::new().unwrap().set_pq_bits(4).set_pq_dim(8));

        // make sure the setters actually updated internal representation on the c-struct
        unsafe {
            assert_eq!((*params.0).graph_degree, 16);
            assert_eq!((*params.0).intermediate_graph_degree, 128);
            assert_eq!((*params.0).build_algo, BuildAlgo::NN_DESCENT);
            assert_eq!((*params.0).nn_descent_niter, 10);
            assert_eq!((*(*params.0).compression).pq_dim, 8);
            assert_eq!((*(*params.0).compression).pq_bits, 4);
        }
    }
}
