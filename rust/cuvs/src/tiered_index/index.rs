/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::io::{Write, stderr};

use crate::dlpack::{AsDlTensor, AsDlTensorMut, DLTensorView};
use crate::error::{Error, Result, check_cuvs};
use crate::resources::Resources;
use crate::tiered_index::{AnnAlgo, IndexParams, SearchParams};

/// Tiered ANN Index.
///
/// A tiered index couples a brute-force tier that absorbs incremental inserts
/// with an ANN tier (CAGRA by default). Vectors added via [`Index::extend`]
/// land in the brute-force tier and are immediately searchable, even before the
/// ANN tier has been (re)built — this is the defining behavior of the tiered
/// index.
///
/// The C API offers no serialize/deserialize for the tiered index, so this
/// wrapper does not expose persistence (and therefore takes no filesystem
/// paths).
#[derive(Debug)]
pub struct Index {
    handle: ffi::cuvsTieredIndex_t,
    /// The ANN backend the index was built with, used to validate that search
    /// params match the backend (the C search API reinterprets an opaque
    /// `void*` per this algo).
    algo: AnnAlgo,
}

impl Index {
    /// Builds a new tiered Index over `dataset` for efficient search.
    ///
    /// `dataset` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]. It is copied into the index, so the caller may free it
    /// once this call returns (hence `Index` carries no lifetime).
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index (backend + ANN params)
    /// * `dataset` - The dataset to index
    pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let handle = Index::create_handle()?;
        unsafe {
            check_cuvs(ffi::cuvsTieredIndexBuild(
                res.0,
                params.as_ptr(),
                dataset.to_c().as_mut_ptr(),
                handle,
            ))?;
        }
        // Capture the backend so search() can reject mismatched params.
        Ok(Index { handle, algo: params.algo() })
    }

    /// Creates a new empty index handle.
    ///
    /// Private: [`Index::build`] is the only public constructor so that every
    /// `Index` carries the backend algo captured from its build params.
    fn create_handle() -> Result<ffi::cuvsTieredIndex_t> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsTieredIndex_t>::uninit();
            check_cuvs(ffi::cuvsTieredIndexCreate(index.as_mut_ptr()))?;
            Ok(index.assume_init())
        }
    }

    /// Extends the index with new vectors.
    ///
    /// The new vectors are added to the brute-force tier and become immediately
    /// searchable. If `create_ann_index_on_extend` was set and the incremental
    /// tier now exceeds `min_ann_rows`, the ANN tier is (re)built.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `new_vectors` - A row-major matrix on the host or device implementing
    ///   [`AsDlTensor`], to add to the index
    pub fn extend<T>(&self, res: &Resources, new_vectors: &T) -> Result<()>
    where
        T: AsDlTensor + ?Sized,
    {
        let new_vectors = new_vectors.as_dl_tensor()?;
        unsafe {
            check_cuvs(ffi::cuvsTieredIndexExtend(
                res.0,
                new_vectors.to_c().as_mut_ptr(),
                self.handle,
            ))
        }
    }

    /// Performs an Approximate Nearest Neighbors search on the Index.
    ///
    /// `params` must match the ANN backend the index was built with (e.g.
    /// [`SearchParams::Cagra`] for a CAGRA-backed index); a mismatch returns
    /// [`crate::error::Error::InvalidArgument`] rather than risking the
    /// undefined behavior of the C API reinterpreting the wrong struct.
    ///
    /// `queries`, `neighbors`, and `distances` must reside in device memory and
    /// implement [`AsDlTensor`] / [`AsDlTensorMut`];
    /// `neighbors` and `distances` are written in place.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Search parameters for the ANN backend
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
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
        let no_filter = ffi::cuvsFilter { addr: 0, type_: ffi::cuvsFilterType::NO_FILTER };
        self.search_impl(res, params, &queries, &neighbors, &distances, no_filter)
    }

    /// Performs an Approximate Nearest Neighbors search with a bitset prefilter.
    ///
    /// Like [`search`](Self::search), but accepts a bitset filter to exclude
    /// vectors from the result set. `params` must match the index's backend;
    /// a mismatch returns [`crate::error::Error::InvalidArgument`].
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Search parameters for the ANN backend
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    /// * `bitset` - A 1-D `uint32` device tensor with `ceil(n_rows / 32)` elements.
    ///   Each bit corresponds to a dataset row: bit 1 = include, bit 0 = exclude.
    pub fn search_with_filter<Q, N, D, B>(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &Q,
        neighbors: &mut N,
        distances: &mut D,
        bitset: &B,
    ) -> Result<()>
    where
        Q: AsDlTensor + ?Sized,
        N: AsDlTensorMut + ?Sized,
        D: AsDlTensorMut + ?Sized,
        B: AsDlTensor + ?Sized,
    {
        let queries = queries.as_dl_tensor()?;
        let neighbors = neighbors.as_dl_tensor_mut()?;
        let distances = distances.as_dl_tensor_mut()?;
        let bitset = bitset.as_dl_tensor()?;
        // The bitset pointer is cast to `usize` and stored in `prefilter`, then
        // read by the search call, so its `ManagedTensorRef` must outlive both.
        let mut bitset_c = bitset.to_c();
        let prefilter = ffi::cuvsFilter {
            addr: bitset_c.as_mut_ptr() as usize,
            type_: ffi::cuvsFilterType::BITSET,
        };
        self.search_impl(res, params, &queries, &neighbors, &distances, prefilter)
    }

    /// Shared search path for the filtered and unfiltered variants.
    ///
    /// Validates that `params` matches the index's build-time backend before
    /// handing the opaque pointer to the C API (which reinterprets it per the
    /// index's algo — a mismatch would be undefined behavior).
    fn search_impl(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &DLTensorView,
        neighbors: &DLTensorView,
        distances: &DLTensorView,
        prefilter: ffi::cuvsFilter,
    ) -> Result<()> {
        if params.algo() != self.algo {
            return Err(Error::InvalidArgument(format!(
                "searched with {:?} params but index was built with {:?}",
                params.algo(),
                self.algo,
            )));
        }
        unsafe {
            check_cuvs(ffi::cuvsTieredIndexSearch(
                res.0,
                // The C API takes the backend's search params as an opaque
                // void*; the variant is validated above to match the index's
                // build-time backend.
                params.as_void_ptr(),
                self.handle,
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
                prefilter,
            ))
        }
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsTieredIndexDestroy(self.handle) }) {
            write!(stderr(), "failed to call cuvsTieredIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::DeviceTensor;
    use crate::tiered_index::AnnAlgo;
    use crate::{cagra, ivf_flat};
    use ndarray::s;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    fn default_params() -> IndexParams {
        // Keep min_ann_rows low so the ANN tier (CAGRA) is exercised by the
        // build, while leaving room for extend to land in the brute-force tier.
        IndexParams::new()
            .unwrap()
            .set_algo(AnnAlgo::CUVS_TIERED_INDEX_ALGO_CAGRA)
            .set_min_ann_rows(128)
            .set_create_ann_index_on_extend(true)
    }

    /// (a) Build with an initial dataset and confirm each query finds itself.
    #[test]
    fn test_tiered_build_and_search() {
        let res = Resources::new().unwrap();

        let n_datapoints = 1024;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let index = Index::build(&res, &default_params(), &dataset_device)
            .expect("failed to build tiered index");

        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();

        let k = 10;
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::Cagra(cagra::SearchParams::new().unwrap());
        index.search(&res, &search_params, &queries, &mut neighbors, &mut distances).unwrap();

        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
        for i in 0..n_queries {
            assert_eq!(neighbors_host[[i, 0]], i as i64, "query {i} should find itself");
        }
    }

    /// A non-CAGRA backend must work end-to-end, proving the
    /// `SearchParams::algo()` / `as_void_ptr()` mapping for IVF-Flat is
    /// correct (the mismatch test alone only proves rejection).
    #[test]
    fn test_tiered_ivf_flat_backend_search() {
        let res = Resources::new().unwrap();

        let n_datapoints = 1024;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let params = IndexParams::new()
            .unwrap()
            .set_algo(AnnAlgo::CUVS_TIERED_INDEX_ALGO_IVF_FLAT)
            .set_min_ann_rows(128)
            .set_ivf_flat_params(crate::ivf_flat::IndexParams::new().unwrap());
        let index = Index::build(&res, &params, &dataset_device)
            .expect("failed to build IVF-Flat-backed tiered index");

        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();

        let k = 10;
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::IvfFlat(crate::ivf_flat::SearchParams::new().unwrap());
        index.search(&res, &search_params, &queries, &mut neighbors, &mut distances).unwrap();

        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
        for i in 0..n_queries {
            assert_eq!(neighbors_host[[i, 0]], i as i64, "query {i} should find itself");
        }
    }

    /// (b) THE KEY TEST: vectors added via extend after build must be
    /// immediately findable — the entire point of the tiered index.
    #[test]
    fn test_tiered_extend_visibility() {
        let res = Resources::new().unwrap();

        let n_features = 16;
        let n_initial = 512;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_initial, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let index = Index::build(&res, &default_params(), &dataset_device)
            .expect("failed to build tiered index");

        // New vectors NOT in the original dataset.
        let n_new = 8;
        let new_vectors =
            ndarray::Array::<f32, _>::random((n_new, n_features), Uniform::new(10., 11.0).unwrap());
        let new_device = DeviceTensor::from_host(&res, &new_vectors).unwrap();
        index.extend(&res, &new_device).expect("extend failed");

        // Query with the new vectors themselves. Their ids are appended after
        // the initial dataset, so neighbor[i][0] should be n_initial + i.
        let queries = DeviceTensor::from_host(&res, &new_vectors).unwrap();
        let k = 5;
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_new, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_new, k]).unwrap();
        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_new, k));
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_new, k]).unwrap();

        let search_params = SearchParams::Cagra(cagra::SearchParams::new().unwrap());
        index.search(&res, &search_params, &queries, &mut neighbors, &mut distances).unwrap();

        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
        distances.copy_to_host(&res, &mut distances_host).unwrap();
        for i in 0..n_new {
            assert_eq!(
                neighbors_host[[i, 0]],
                (n_initial + i) as i64,
                "extended vector {i} must be immediately findable as its own nearest neighbor"
            );
            // Self-distance should be ~zero up to float32 rounding.
            assert!(
                distances_host[[i, 0]] < 1e-2,
                "extended vector {i} self-distance {} should be ~0",
                distances_host[[i, 0]]
            );
        }
    }

    /// (c) Repeated extends each remain searchable.
    #[test]
    fn test_tiered_repeated_extends() {
        let res = Resources::new().unwrap();

        let n_features = 16;
        let n_initial = 512;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_initial, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let index = Index::build(&res, &default_params(), &dataset_device)
            .expect("failed to build tiered index");

        let search_params = SearchParams::Cagra(cagra::SearchParams::new().unwrap());
        let k = 8;
        let n_batch = 4;
        let mut total = n_initial;

        for round in 0..3 {
            // Use a distinct value range per round so each batch is far from the
            // [0,1] base cluster and from the other rounds.
            let lo = 20.0 + round as f32 * 10.0;
            let new_vectors = ndarray::Array::<f32, _>::random(
                (n_batch, n_features),
                Uniform::new(lo, lo + 1.0).unwrap(),
            );
            let new_device = DeviceTensor::from_host(&res, &new_vectors).unwrap();
            index.extend(&res, &new_device).expect("extend failed");

            let queries = DeviceTensor::from_host(&res, &new_vectors).unwrap();
            let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_batch, k));
            let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_batch, k]).unwrap();
            let mut distances_host = ndarray::Array::<f32, _>::zeros((n_batch, k));
            let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_batch, k]).unwrap();

            index.search(&res, &search_params, &queries, &mut neighbors, &mut distances).unwrap();
            neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
            distances.copy_to_host(&res, &mut distances_host).unwrap();
            // Each just-extended vector must be immediately findable: its own id
            // appears in the top-k with a near-zero self-distance. We assert
            // top-k membership rather than exact rank-0 to stay robust against
            // the ANN tier's approximate recall after an on-extend rebuild.
            for i in 0..n_batch {
                let want = (total + i) as i64;
                let pos = (0..k).find(|&j| neighbors_host[[i, j]] == want);
                let pos = pos.unwrap_or_else(|| {
                    panic!(
                        "round {round}: extended vector {i} (id {want}) not found in top-{k}: {:?}",
                        neighbors_host.row(i)
                    )
                });
                // Self-distance is ~0 up to float32 rounding (a handful of
                // ULPs across the feature dimension); real neighbors in other
                // value ranges are orders of magnitude farther.
                assert!(
                    distances_host[[i, pos]] < 1e-2,
                    "round {round}: extended vector {i} self-distance {} should be ~0",
                    distances_host[[i, pos]]
                );
            }
            total += n_batch;
        }
    }

    /// (d) Filtered search: a bitset prefilter that excludes a query's own id
    /// must keep that id out of the result set.
    #[test]
    fn test_tiered_filtered_search() {
        let res = Resources::new().unwrap();

        let n_features = 16;
        let n_datapoints = 1024;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let index = Index::build(&res, &default_params(), &dataset_device)
            .expect("failed to build tiered index");

        // Build a bitset over n_datapoints with all bits set (1 = keep), then
        // clear bit 0 so query 0 cannot return itself. cuvs bitsets are u32
        // words, LSB-first.
        let n_words = n_datapoints.div_ceil(32);
        let mut bitset_host = ndarray::Array::<u32, _>::from_elem(n_words, u32::MAX);
        bitset_host[0] &= !1u32; // clear bit 0 -> exclude id 0
        let bitset = DeviceTensor::from_host(&res, &bitset_host).unwrap();

        let n_queries = 1;
        let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();
        let k = 5;
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::Cagra(cagra::SearchParams::new().unwrap());
        index
            .search_with_filter(
                &res,
                &search_params,
                &queries,
                &mut neighbors,
                &mut distances,
                &bitset,
            )
            .unwrap();

        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
        // id 0 was filtered out, so it must not appear among the neighbors.
        for j in 0..k {
            assert_ne!(neighbors_host[[0, j]], 0, "filtered id 0 must not be returned");
        }
    }

    /// (e) Backend mismatch: searching a CAGRA-backed index with IVF-Flat
    /// search params must be rejected before reaching the C API (which would
    /// otherwise reinterpret the wrong struct — undefined behavior).
    #[test]
    fn test_search_params_backend_mismatch() {
        let res = Resources::new().unwrap();

        let n_features = 16;
        let n_datapoints = 1024;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        // CAGRA-backed index (default_params sets CUVS_TIERED_INDEX_ALGO_CAGRA).
        let index = Index::build(&res, &default_params(), &dataset_device)
            .expect("failed to build tiered index");

        let n_queries = 1;
        let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();
        let k = 5;
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        // Wrong-backend params: IVF-Flat against a CAGRA index.
        let search_params = SearchParams::IvfFlat(ivf_flat::SearchParams::new().unwrap());
        let err = index
            .search(&res, &search_params, &queries, &mut neighbors, &mut distances)
            .expect_err("mismatched search params must be rejected");

        match err {
            Error::InvalidArgument(msg) => {
                assert!(
                    msg.contains("IVF_FLAT") && msg.contains("CAGRA"),
                    "error should mention both the searched and built backends: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
