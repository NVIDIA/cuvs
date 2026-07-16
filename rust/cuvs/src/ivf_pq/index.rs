/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::io::{Write, stderr};

use crate::dlpack::{AsDlTensor, AsDlTensorMut};
use crate::error::{Result, check_cuvs};
use crate::ivf_pq::{IndexParams, SearchParams};
use crate::resources::Resources;

/// Ivf-Pq ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsIvfPqIndex_t);

impl Index {
    /// Builds an IVF-PQ index over `dataset` for efficient search.
    ///
    /// `dataset` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]. It is copied (and quantized) into
    /// the index, so the caller may free it once this call returns (hence
    /// `Index` carries no lifetime).
    ///
    /// Supported dataset/query dtypes in the current C-backed implementation are
    /// `f32`, `f16`, `i8`, and `u8`.
    pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsIvfPqBuild(res.0, params.0, dataset.to_c().as_mut_ptr(), index.0))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsIvfPqIndex_t>::uninit();
            check_cuvs(ffi::cuvsIvfPqIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Searches the index for the `k` nearest neighbors of each query.
    ///
    /// `queries`, `neighbors`, and `distances` must reside in device memory and
    /// implement [`AsDlTensor`] /
    /// [`AsDlTensorMut`]. `neighbors` receives the
    /// neighbor indices and `distances` their distances; both are written in
    /// place.
    pub fn search<Q, N, D>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &Q,
        neighbors: &mut N,
        distances: &mut D,
    ) -> Result<()>
    where
        Q: AsDlTensor + ?Sized,
        N: AsDlTensorMut + ?Sized,
        D: AsDlTensorMut + ?Sized,
    {
        let queries = queries.as_dl_tensor()?;
        let neighbors = neighbors.as_dl_tensor_mut()?;
        let distances = distances.as_dl_tensor_mut()?;
        unsafe {
            check_cuvs(ffi::cuvsIvfPqSearch(
                res.0,
                params.0,
                self.0,
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
            ))
        }
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfPqIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsIvfPqIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::DeviceTensor;
    use ndarray::s;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_ivf_pq() {
        let build_params = IndexParams::new().unwrap().set_n_lists(64);

        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 1024;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        // build the ivf-pq index
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to create ivf-pq index");

        // use the first 4 points from the dataset as queries : will test that we get them back
        // as their own nearest neighbor
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]).to_owned();

        let k = 10;

        // Ivf-Pq search API requires queries and outputs to be on device memory
        // copy query data over, and allocate new device memory for the distances/ neighbors
        // outputs
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::new().unwrap();

        index.search(&res, &search_params, &queries, &mut neighbors, &mut distances).unwrap();

        // Copy back to host memory
        distances.copy_to_host(&res, &mut distances_host).unwrap();
        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();

        // nearest neighbors should be themselves, since queries are from the
        // dataset
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    /// Test that an index can be searched multiple times without rebuilding.
    /// This validates that search() takes &self instead of self.
    #[test]
    fn test_ivf_pq_multiple_searches() {
        let build_params = IndexParams::new().unwrap().set_n_lists(64);
        let res = Resources::new().unwrap();

        // Create a random dataset
        let n_datapoints = 1024;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        // Build the index once
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to create ivf-pq index");

        let search_params = SearchParams::new().unwrap();
        let k = 5;

        // Perform multiple searches on the same index
        for search_iter in 0..3 {
            let n_queries = 4;
            let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
            let queries = DeviceTensor::from_host(&res, &queries).unwrap();

            let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
            let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();

            let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
            let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

            // This should work on every iteration because search() takes &self
            index
                .search(&res, &search_params, &queries, &mut neighbors, &mut distances)
                .expect(&format!("search iteration {} failed", search_iter));

            // Copy back to host memory
            distances.copy_to_host(&res, &mut distances_host).unwrap();
            neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();

            // Verify results are consistent
            assert_eq!(
                neighbors_host[[0, 0]],
                0,
                "iteration {}: first query should find itself",
                search_iter
            );
        }
    }
}
