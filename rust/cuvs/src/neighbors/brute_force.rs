/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Brute-force (exact) k-NN.
//!
//! Build an [`Index`] over a dataset, then [`search`](Index::search) it with
//! device-resident queries and output buffers. Tensors are borrowed through the
//! `AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
//! module for the tensor model and `examples/cagra.rs` for the same
//! build/search workflow.

use std::io::{Write, stderr};
use std::marker::PhantomData;

use crate::distance::DistanceType;
use crate::dlpack::{AsDlTensor, AsDlTensorMut, DLPackError, DLTensorView, DLTensorViewMut};
use crate::error::{LibraryError, check_cuvs};
pub use crate::neighbors::filters::SearchFilter;
use crate::neighbors::filters::with_filter;
use crate::resources::Resources;

type Result<T> = std::result::Result<T, BruteForceError>;

/// Error type for brute-force operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BruteForceError {
    /// The cuVS C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
}

/// Brute-force KNN index.
#[derive(Debug)]
pub struct Index<'d> {
    inner: ffi::cuvsBruteForceIndex_t,
    // cuVS brute_force::index stores a non-owning view into the dataset.
    // Keep the Rust borrow alive for as long as the C++ index may read it.
    _dataset: PhantomData<&'d ()>,
}

impl<'d> Index<'d> {
    /// Builds a brute-force index over `dataset` for exact k-NN search.
    ///
    /// `metric` selects the distance (use [`DistanceType::LpUnexpanded`] to set
    /// the Minkowski exponent `p`). `dataset` is a row-major matrix on the host
    /// or device implementing [`AsDlTensor`]; the C++ index keeps a non-owning
    /// view of it, so the returned [`Index`] borrows it for `'d` and cannot
    /// outlive it.
    pub fn build<T>(res: &Resources, metric: DistanceType, dataset: &'d T) -> Result<Index<'d>>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::create_handle()?;
        unsafe {
            check_cuvs(ffi::cuvsBruteForceBuild(
                res.handle(),
                dataset.to_c().as_mut_ptr(),
                metric.into(),
                metric.metric_arg(),
                index.inner,
            ))?;
        }
        Ok(index)
    }

    fn create_handle() -> Result<Index<'d>> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsBruteForceIndex_t>::uninit();
            check_cuvs(ffi::cuvsBruteForceIndexCreate(index.as_mut_ptr()))?;
            Ok(Index { inner: index.assume_init(), _dataset: PhantomData })
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
        self.search_impl(res, &queries, &neighbors, &distances, None)
    }

    /// Searches the index using a row bitset or per-query bitmap filter.
    pub fn search_filtered<Q, N, D>(
        &self,
        res: &Resources,
        queries: &Q,
        neighbors: &mut N,
        distances: &mut D,
        filter: &SearchFilter<'_>,
    ) -> Result<()>
    where
        Q: AsDlTensor + ?Sized,
        N: AsDlTensorMut + ?Sized,
        D: AsDlTensorMut + ?Sized,
    {
        let queries = queries.as_dl_tensor()?;
        let neighbors = neighbors.as_dl_tensor_mut()?;
        let distances = distances.as_dl_tensor_mut()?;
        self.search_impl(res, &queries, &neighbors, &distances, Some(filter))
    }

    fn search_impl(
        &self,
        res: &Resources,
        queries: &DLTensorView<'_>,
        neighbors: &DLTensorViewMut<'_>,
        distances: &DLTensorViewMut<'_>,
        filter: Option<&SearchFilter<'_>>,
    ) -> Result<()> {
        with_filter(filter, |prefilter| {
            check_cuvs(unsafe {
                ffi::cuvsBruteForceSearch(
                    res.handle(),
                    self.inner,
                    queries.to_c().as_mut_ptr(),
                    neighbors.to_c().as_mut_ptr(),
                    distances.to_c().as_mut_ptr(),
                    prefilter,
                )
            })?;
            Ok(())
        })
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsBruteForceIndexDestroy(self.inner) }) {
            write!(stderr(), "failed to call bruteForceIndexDestroy {:?}", e)
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

    fn test_bfknn(metric: DistanceType) {
        let res = Resources::new().unwrap();

        // Create a new random dataset to index.
        let n_datapoints = 16;
        let n_features = 8;
        let dataset_host = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset = DeviceTensor::from_host(&res, &dataset_host).unwrap();

        let index =
            Index::build(&res, metric, &dataset).expect("failed to create brute force index");

        res.sync_stream().unwrap();

        // Use the first 4 points from the dataset as queries: each should get
        // itself back as its own nearest neighbor.
        let n_queries = 4;
        let queries = dataset_host.slice(s![0..n_queries, ..]).to_owned();
        let k = 4;

        let queries = DeviceTensor::from_host(&res, &queries).unwrap();
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        index.search(&res, &queries, &mut neighbors, &mut distances).unwrap();

        distances.copy_to_host(&res, &mut distances_host).unwrap();
        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
        res.sync_stream().unwrap();

        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    #[test]
    fn test_l2() {
        test_bfknn(DistanceType::L2Expanded);
    }
}
