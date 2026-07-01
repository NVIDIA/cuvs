/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::io::{Write, stderr};

use super::{IndexParams, IvfPqError, SearchParams};
use crate::dlpack::{AsDlTensor, AsDlTensorMut};
use crate::error::check_cuvs;
use crate::resources::Resources;

type Result<T> = std::result::Result<T, IvfPqError>;

/// IVF-PQ ANN index.
#[derive(Debug)]
pub struct Index(ffi::cuvsIvfPqIndex_t);

impl Index {
    /// Builds an IVF-PQ index over `dataset` for compressed, efficient search.
    ///
    /// `dataset` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]. It is copied into the index, so the caller may free it
    /// once this call returns (hence `Index` carries no lifetime).
    pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsIvfPqBuild(
                res.handle(),
                params.handle(),
                dataset.to_c().as_mut_ptr(),
                index.0,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index.
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
    /// implement [`AsDlTensor`] / [`AsDlTensorMut`]. `neighbors` receives the
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
        check_cuvs(unsafe {
            ffi::cuvsIvfPqSearch(
                res.handle(),
                params.handle(),
                self.0,
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
            )
        })?;
        Ok(())
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
        let build_params = IndexParams::builder().n_lists(64).build().unwrap();
        let res = Resources::new().unwrap();

        let n_datapoints = 1024;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to create ivf-pq index");

        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
        let k = 10;

        let queries = DeviceTensor::from_host(&res, &queries).unwrap();
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::builder().build().unwrap();

        index.search(&res, &search_params, &queries, &mut neighbors, &mut distances).unwrap();

        distances.copy_to_host(&res, &mut distances_host).unwrap();
        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();

        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    /// Searching the same index multiple times validates that `search` takes
    /// `&self` rather than consuming the index.
    #[test]
    fn test_ivf_pq_multiple_searches() {
        let build_params = IndexParams::builder().n_lists(64).build().unwrap();
        let res = Resources::new().unwrap();

        let n_datapoints = 1024;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to create ivf-pq index");

        let search_params = SearchParams::builder().build().unwrap();
        let k = 5;

        for _ in 0..3 {
            let n_queries = 4;
            let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
            let queries = DeviceTensor::from_host(&res, &queries).unwrap();

            let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
            let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
            let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

            index
                .search(&res, &search_params, &queries, &mut neighbors, &mut distances)
                .expect("search failed");

            neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
            assert_eq!(neighbors_host[[0, 0]], 0, "first query should find itself");
        }
    }
}
