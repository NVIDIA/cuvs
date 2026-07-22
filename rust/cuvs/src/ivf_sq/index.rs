/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{Write, stderr};
use std::path::Path;

use crate::dlpack::{AsDlTensor, AsDlTensorMut};
use crate::error::{Error, Result, check_cuvs};
use crate::ivf_sq::{IndexParams, SearchParams};
use crate::resources::Resources;

/// IVF-SQ ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsIvfSqIndex_t);

/// Convert a filesystem path into a `CString` suitable for the cuVS C API,
/// returning `Error::InvalidArgument` instead of panicking for paths that are
/// not valid UTF-8 or that contain an interior NUL byte.
fn path_to_cstring(path: &Path) -> Result<CString> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("path is not valid UTF-8: {path:?}")))?;
    CString::new(path_str)
        .map_err(|e| Error::InvalidArgument(format!("path contains an interior NUL byte: {e}")))
}

impl Index {
    /// Builds an IVF-SQ index over `dataset` for efficient search.
    ///
    /// `dataset` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]. It is copied (and quantized) into the index, so the
    /// caller may free it once this call returns (hence `Index` carries no
    /// lifetime).
    pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsIvfSqBuild(res.0, params.0, dataset.to_c().as_mut_ptr(), index.0))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsIvfSqIndex_t>::uninit();
            check_cuvs(ffi::cuvsIvfSqIndexCreate(index.as_mut_ptr()))?;
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
            let prefilter = ffi::cuvsFilter { addr: 0, type_: ffi::cuvsFilterType::NO_FILTER };

            check_cuvs(ffi::cuvsIvfSqSearch(
                res.0,
                params.0,
                self.0,
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
                prefilter,
            ))
        }
    }

    /// Perform a filtered Approximate Nearest Neighbors search on the Index
    ///
    /// Like [`search`](Self::search), but applies a bitset filter to exclude
    /// vectors from the candidate set. Filtered vectors are never returned,
    /// giving better recall than post-filtering.
    ///
    /// `queries`, `neighbors`, and `distances` are as in [`search`](Self::search).
    /// `bitset` is a 1-D `uint32` device tensor of `ceil(n_rows / 32)` elements,
    /// where each bit corresponds to a dataset row: bit 1 = include, bit 0 =
    /// exclude.
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
        // The bitset pointer is cast to `usize` and stored in `prefilter`, then read
        // by the search call, so its `ManagedTensorRef` must outlive both.
        // Hence we keep it bound instead of chaining `to_c().as_mut_ptr()`.
        let mut bitset_c = bitset.to_c();
        unsafe {
            let prefilter = ffi::cuvsFilter {
                addr: bitset_c.as_mut_ptr() as usize,
                type_: ffi::cuvsFilterType::BITSET,
            };

            check_cuvs(ffi::cuvsIvfSqSearch(
                res.0,
                params.0,
                self.0,
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
                prefilter,
            ))
        }
    }

    /// Extend the index with new vectors.
    ///
    /// `new_vectors` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]. `new_indices` is an optional 1-D tensor of indices for
    /// the new vectors; if `None`, a continuous range `[0..n_rows)` is implied,
    /// which is only valid when the index is currently empty.
    pub fn extend<V>(
        &mut self,
        res: &Resources,
        new_vectors: &V,
        new_indices: Option<&dyn AsDlTensor>,
    ) -> Result<()>
    where
        V: AsDlTensor + ?Sized,
    {
        let new_vectors = new_vectors.as_dl_tensor()?;
        // Bind the optional index view, then its `ManagedTensorRef`, so both
        // outlive the FFI call whose pointer borrows them.
        let new_indices = new_indices.map(|t| t.as_dl_tensor()).transpose()?;
        let mut new_indices_c = new_indices.as_ref().map(|v| v.to_c());
        let new_indices_ptr =
            new_indices_c.as_mut().map_or(std::ptr::null_mut(), |c| c.as_mut_ptr());
        unsafe {
            check_cuvs(ffi::cuvsIvfSqExtend(
                res.0,
                new_vectors.to_c().as_mut_ptr(),
                new_indices_ptr,
                self.0,
            ))
        }
    }

    // `cuvsIvfSqIndexGetCenters` is deliberately not wrapped, matching the
    // cagra and ivf_pq Rust modules: it fills a caller-allocated
    // DLManagedTensor, which has no ergonomic safe-Rust shape yet.

    /// The number of clusters / inverted lists in the index.
    pub fn n_lists(&self) -> Result<i64> {
        let mut n_lists: i64 = 0;
        unsafe {
            check_cuvs(ffi::cuvsIvfSqIndexGetNLists(self.0, &mut n_lists))?;
        }
        Ok(n_lists)
    }

    /// The dimensionality of the vectors stored in the index.
    pub fn dim(&self) -> Result<i64> {
        let mut dim: i64 = 0;
        unsafe {
            check_cuvs(ffi::cuvsIvfSqIndexGetDim(self.0, &mut dim))?;
        }
        Ok(dim)
    }

    /// The number of vectors stored in the index.
    pub fn size(&self) -> Result<i64> {
        let mut size: i64 = 0;
        unsafe {
            check_cuvs(ffi::cuvsIvfSqIndexGetSize(self.0, &mut size))?;
        }
        Ok(size)
    }

    /// Save the IVF-SQ index to file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    pub fn serialize<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        unsafe { check_cuvs(ffi::cuvsIvfSqSerialize(res.0, c_filename.as_ptr(), self.0)) }
    }

    /// Load an IVF-SQ index from file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The path of the file that stores the index
    pub fn deserialize<P: AsRef<Path>>(res: &Resources, filename: P) -> Result<Index> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsIvfSqDeserialize(res.0, c_filename.as_ptr(), index.0))?;
        }
        Ok(index)
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfSqIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsIvfSqIndexDestroy {:?}", e)
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

    const N_DATAPOINTS: usize = 1024;
    const N_FEATURES: usize = 16;

    /// Build a small random dataset and an IVF-SQ index over it.
    fn build_test_index(
        res: &Resources,
        build_params: &IndexParams,
    ) -> (ndarray::Array2<f32>, Index) {
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(res, &dataset).unwrap();
        let index =
            Index::build(res, build_params, &dataset_device).expect("failed to build ivf-sq index");
        (dataset, index)
    }

    /// Search the first `n_queries` rows of `dataset` against `index` and assert
    /// each query finds itself as the top-1 neighbor. IVF-SQ search requires
    /// queries and outputs to live in device memory.
    fn search_and_verify_self_neighbors(
        res: &Resources,
        index: &Index,
        dataset: &ndarray::Array2<f32>,
        n_queries: usize,
        k: usize,
    ) {
        let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
        let queries = DeviceTensor::from_host(res, &queries).unwrap();

        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(res, &[n_queries, k]).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let mut distances = DeviceTensor::<f32>::zeros(res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::new().unwrap();
        index
            .search(res, &search_params, &queries, &mut neighbors, &mut distances)
            .expect("search failed");

        distances.copy_to_host(res, &mut distances_host).unwrap();
        neighbors.copy_to_host(res, &mut neighbors_host).unwrap();

        for i in 0..n_queries {
            assert_eq!(
                neighbors_host[[i, 0]],
                i as i64,
                "query {i} should be its own nearest neighbor"
            );
        }
    }

    #[test]
    fn test_ivf_sq() {
        let build_params = IndexParams::new().unwrap().set_n_lists(64);
        let res = Resources::new().unwrap();
        let (dataset, index) = build_test_index(&res, &build_params);
        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);
    }

    /// Test that an index can be searched multiple times without rebuilding.
    /// This validates that `search()` takes `&self` instead of `self`.
    #[test]
    fn test_ivf_sq_multiple_searches() {
        let build_params = IndexParams::new().unwrap().set_n_lists(64);
        let res = Resources::new().unwrap();
        let (dataset, index) = build_test_index(&res, &build_params);

        for _ in 0..3 {
            search_and_verify_self_neighbors(&res, &index, &dataset, 4, 5);
        }
    }

    /// Test the index introspection getters against the build parameters.
    #[test]
    fn test_ivf_sq_index_getters() {
        let build_params = IndexParams::new().unwrap().set_n_lists(64);
        let res = Resources::new().unwrap();
        let (_dataset, index) = build_test_index(&res, &build_params);

        assert_eq!(index.n_lists().unwrap(), 64);
        assert_eq!(index.dim().unwrap(), N_FEATURES as i64);
        assert_eq!(index.size().unwrap(), N_DATAPOINTS as i64);
    }

    /// Test extending an index that was built with `add_data_on_build == false`.
    #[test]
    fn test_ivf_sq_extend() {
        let build_params = IndexParams::new().unwrap().set_n_lists(64).set_add_data_on_build(false);
        let res = Resources::new().unwrap();

        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        // Build only trains the quantizer/clustering; the index is left empty.
        let mut index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to build ivf-sq index");
        assert_eq!(index.size().unwrap(), 0, "index should be empty before extend");

        // Populate the index with the dataset using a continuous index range.
        let extend_vectors = DeviceTensor::from_host(&res, &dataset).unwrap();
        index.extend(&res, &extend_vectors, None).expect("failed to extend ivf-sq index");
        assert_eq!(index.size().unwrap(), N_DATAPOINTS as i64);

        // The populated index should now find each query as its own neighbor.
        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);
    }

    /// Test bitset-filtered search: exclude odd-indexed rows, verify they don't appear.
    #[test]
    fn test_ivf_sq_search_with_filter() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap().set_n_lists(64);
        let (dataset, index) = build_test_index(&res, &build_params);

        // Build a bitset that includes only even-indexed rows.
        let n_words = N_DATAPOINTS.div_ceil(32);
        let mut bitset_host = ndarray::Array::<u32, _>::zeros(ndarray::Ix1(n_words));
        for i in 0..N_DATAPOINTS {
            if i % 2 == 0 {
                bitset_host[i / 32] |= 1u32 << (i % 32);
            }
        }
        let bitset = DeviceTensor::from_host(&res, &bitset_host).unwrap();

        // Query with the first 4 even-indexed rows.
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries * 2;2, ..]).to_owned(); // rows 0, 2, 4, 6
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();

        let k = 10;
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::new().unwrap();
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

        // All returned neighbors must be even-indexed (odd rows are filtered out).
        for q in 0..n_queries {
            for n in 0..k {
                let neighbor_id = neighbors_host[[q, n]];
                assert_eq!(
                    neighbor_id % 2,
                    0,
                    "query {q}, neighbor {n}: got odd index {neighbor_id}, expected only even"
                );
            }
        }

        // First query (row 0) should find itself as the nearest neighbor.
        assert_eq!(neighbors_host[[0, 0]], 0);
    }

    #[test]
    fn test_ivf_sq_serialize_deserialize() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap().set_n_lists(64);
        let (dataset, index) = build_test_index(&res, &build_params);

        let unique = format!(
            "test_ivf_sq_index_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        );
        let filepath = std::env::temp_dir().join(unique);
        index.serialize(&res, &filepath).expect("failed to serialize ivf-sq index");

        assert!(filepath.exists(), "serialized index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized index file should not be empty"
        );

        let loaded_index =
            Index::deserialize(&res, &filepath).expect("failed to deserialize ivf-sq index");

        // The deserialized index should still find each query as its own
        // nearest neighbor.
        search_and_verify_self_neighbors(&res, &loaded_index, &dataset, 4, 10);

        let _ = std::fs::remove_file(&filepath);
    }

    /// Passing a filename containing an interior NUL byte must surface as an
    /// `InvalidArgument` error rather than panicking inside the serializer.
    #[test]
    fn test_ivf_sq_serialize_rejects_interior_nul() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap().set_n_lists(64);
        let (_dataset, index) = build_test_index(&res, &build_params);

        // `PathBuf::from` on Unix preserves arbitrary bytes, so we can embed a
        // NUL byte in the path and confirm the helper rejects it.
        let bad_path = std::path::PathBuf::from("/tmp/has\0nul.bin");
        let err = index
            .serialize(&res, &bad_path)
            .expect_err("serialize should reject paths with interior NUL");
        assert!(matches!(err, Error::InvalidArgument(_)), "expected InvalidArgument, got {err:?}");
    }
}
