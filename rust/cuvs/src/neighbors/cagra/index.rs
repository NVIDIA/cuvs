/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{Write, stderr};
use std::marker::PhantomData;
use std::path::Path;

use super::{CagraError, IndexParams, SearchParams};
use crate::dlpack::{AsDlTensor, AsDlTensorMut};
use crate::error::check_cuvs;
use crate::resources::Resources;

type Result<T> = std::result::Result<T, CagraError>;

/// A CAGRA approximate nearest neighbor index.
///
/// The lifetime `'d` ties this index to the underlying dataset,
/// passed at construction time. The C library may store a non-owning view
/// of properly aligned device-resident data, so the dataset must outlive
/// the index. When an index is deserialized from disk, the data is
/// self-contained and its lifetime is `'static`.
#[derive(Debug)]
pub struct Index<'d> {
    handle: ffi::cuvsCagraIndex_t,
    _dataset: PhantomData<&'d ()>,
}

/// Convert a filesystem path into a `CString` suitable for the cuVS C API,
/// returning [`CagraError::InvalidPath`] for a path with an interior NUL byte.
fn path_to_cstring(path: &Path) -> Result<CString> {
    Ok(CString::new(path.as_os_str().as_encoded_bytes())?)
}

impl<'d> Index<'d> {
    /// Builds a CAGRA index over `dataset` for efficient search.
    ///
    /// `dataset` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]. The C++ index keeps a non-owning
    /// view of it, so the returned [`Index`] borrows `dataset` for `'d` and
    /// cannot outlive it.
    pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &'d T) -> Result<Index<'d>>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraBuild(
                res.handle(),
                params.handle(),
                dataset.to_c().as_mut_ptr(),
                index.handle,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index<'d>> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsCagraIndex_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexCreate(index.as_mut_ptr()))?;
            Ok(Index { handle: index.assume_init(), _dataset: PhantomData })
        }
    }

    /// Searches the index for the `k` nearest neighbors of each query.
    ///
    /// `queries`, `neighbors`, and `distances` must reside in device memory and
    /// implement [`AsDlTensor`] /
    /// [`AsDlTensorMut`]. `neighbors` (shape
    /// `n_queries × k`) receives the neighbor indices and `distances` their
    /// distances; both are written in place.
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
        let prefilter = ffi::cuvsFilter { addr: 0, type_: ffi::cuvsFilterType::NO_FILTER };
        check_cuvs(unsafe {
            ffi::cuvsCagraSearch(
                res.handle(),
                params.handle(),
                self.handle,
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
                prefilter,
            )
        })?;
        Ok(())
    }

    /// Perform a filtered Approximate Nearest Neighbors search on the Index
    ///
    /// Like [`search`](Self::search), but applies a bitset filter to exclude
    /// vectors during graph traversal. Filtered vectors are never visited, which
    /// gives better recall than post-filtering.
    ///
    /// `queries`, `neighbors`, and `distances` are as in [`search`](Self::search).
    /// `bitset` is a 1-D `uint32` device tensor of `ceil(n_rows / 32)` elements,
    /// where each bit maps to a dataset row (1 = include, 0 = exclude).
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
        let prefilter = ffi::cuvsFilter {
            addr: bitset_c.as_mut_ptr() as usize,
            type_: ffi::cuvsFilterType::BITSET,
        };
        check_cuvs(unsafe {
            ffi::cuvsCagraSearch(
                res.handle(),
                params.handle(),
                self.handle,
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
                prefilter,
            )
        })?;
        Ok(())
    }

    /// Save the CAGRA index to file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    /// * `include_dataset` - Whether to write out the dataset to the file
    ///
    /// # Example:
    /// ```no_run
    /// use cuvs::Resources;
    /// use cuvs::neighbors::cagra::{Index, IndexParams};
    ///
    /// fn serialize_example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let res = Resources::new()?;
    ///
    ///     // Build an index (using some dataset)
    ///     let build_params = IndexParams::builder().build()?;
    ///     // let index = Index::build(&res, &build_params, &dataset)?;
    ///
    ///     // Save the index to disk (including the dataset)
    ///     // index.serialize(&res, "/path/to/index.bin", true)?;
    ///
    ///     // Later, load the index from disk
    ///     let loaded_index = Index::deserialize(&res, "/path/to/index.bin")?;
    ///
    ///     // The loaded index can be used for search just like the original
    ///     Ok(())
    /// }
    /// ```
    pub fn serialize<P: AsRef<Path>>(
        &self,
        res: &Resources,
        filename: P,
        include_dataset: bool,
    ) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        check_cuvs(unsafe {
            ffi::cuvsCagraSerialize(res.handle(), c_filename.as_ptr(), self.handle, include_dataset)
        })?;
        Ok(())
    }

    /// Save the CAGRA index to file in hnswlib format.
    ///
    /// NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
    /// as the serialization format is not compatible with the original hnswlib.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    pub fn serialize_to_hnswlib<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        check_cuvs(unsafe {
            ffi::cuvsCagraSerializeToHnswlib(res.handle(), c_filename.as_ptr(), self.handle)
        })?;
        Ok(())
    }

    /// Load a CAGRA index from file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The path of the file that stores the index
    pub fn deserialize<P: AsRef<Path>>(res: &Resources, filename: P) -> Result<Index<'static>> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraDeserialize(res.handle(), c_filename.as_ptr(), index.handle))?;
        }
        Ok(index)
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexDestroy(self.handle) }) {
            write!(stderr(), "failed to call cagraIndexDestroy {:?}", e)
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

    const N_DATAPOINTS: usize = 256;
    const N_FEATURES: usize = 16;

    /// Search the first `n_queries` rows of `dataset` against `index` and
    /// assert each query finds itself as the top-1 neighbor. CAGRA search
    /// requires queries and outputs to live in device memory.
    fn search_and_verify_self_neighbors(
        res: &Resources,
        index: &Index<'_>,
        dataset: &ndarray::Array2<f32>,
        n_queries: usize,
        k: usize,
    ) {
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = DeviceTensor::from_host(res, &queries.to_owned()).unwrap();

        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<u32>::zeros(res, &[n_queries, k]).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let mut distances = DeviceTensor::<f32>::zeros(res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::try_new().unwrap();
        index
            .search(res, &search_params, &queries, &mut neighbors, &mut distances)
            .expect("search failed");

        distances.copy_to_host(res, &mut distances_host).unwrap();
        neighbors.copy_to_host(res, &mut neighbors_host).unwrap();

        for i in 0..n_queries {
            assert_eq!(
                neighbors_host[[i, 0]],
                i as u32,
                "query {i} should be its own nearest neighbor"
            );
        }
    }

    fn test_cagra(build_params: IndexParams) {
        let res = Resources::new().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let index =
            Index::build(&res, &build_params, &*dataset).expect("failed to build cagra index");
        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);
    }

    #[test]
    fn test_cagra_index() {
        let build_params = IndexParams::try_new().unwrap();
        test_cagra(build_params);
    }

    #[test]
    fn test_cagra_compression() {
        use crate::neighbors::cagra::CompressionParams;
        let build_params = IndexParams::builder()
            .compression(CompressionParams::builder().build().unwrap())
            .build()
            .unwrap();
        test_cagra(build_params);
    }

    /// Test bitset-filtered search: exclude odd-indexed rows, verify they don't appear.
    #[test]
    fn test_cagra_search_with_filter() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::try_new().unwrap();

        let n_datapoints = 256;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let index =
            Index::build(&res, &build_params, &*dataset).expect("failed to create cagra index");

        // Build a bitset that includes only even-indexed rows
        let n_words = n_datapoints.div_ceil(32);
        let mut bitset_host = ndarray::Array::<u32, _>::zeros(ndarray::Ix1(n_words));
        for i in 0..n_datapoints {
            if i % 2 == 0 {
                bitset_host[i / 32] |= 1u32 << (i % 32);
            }
        }
        let bitset = DeviceTensor::from_host(&res, &bitset_host).unwrap();

        // Query with the first 4 even-indexed rows
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries * 2;2, ..]).to_owned(); // rows 0, 2, 4, 6
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();

        let k = 10;
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<u32>::zeros(&res, &[n_queries, k]).unwrap();
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        let search_params = SearchParams::try_new().unwrap();

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

    /// Test that an index can be searched multiple times without rebuilding.
    /// This validates that `search()` takes `&self` instead of `self`.
    #[test]
    fn test_cagra_multiple_searches() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::try_new().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let index =
            Index::build(&res, &build_params, &*dataset).expect("failed to build cagra index");

        for _ in 0..3 {
            search_and_verify_self_neighbors(&res, &index, &dataset, 4, 5);
        }
    }

    #[test]
    fn test_cagra_serialize_deserialize() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::try_new().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let index =
            Index::build(&res, &build_params, &*dataset).expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index.bin");
        index.serialize(&res, &filepath, true).expect("failed to serialize cagra index");

        assert!(filepath.exists(), "serialized index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized index file should not be empty"
        );

        let loaded_index =
            Index::deserialize(&res, &filepath).expect("failed to deserialize cagra index");

        // The deserialized index should still find each query as its own
        // nearest neighbor.
        search_and_verify_self_neighbors(&res, &loaded_index, &dataset, 4, 10);

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_without_dataset() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::try_new().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let index =
            Index::build(&res, &build_params, &*dataset).expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index_no_dataset.bin");
        index
            .serialize(&res, &filepath, false)
            .expect("failed to serialize cagra index without dataset");

        assert!(filepath.exists(), "serialized index file should exist");

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_to_hnswlib() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::try_new().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let index =
            Index::build(&res, &build_params, &*dataset).expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index_hnsw.bin");
        index
            .serialize_to_hnswlib(&res, &filepath)
            .expect("failed to serialize cagra index to hnswlib format");

        assert!(filepath.exists(), "serialized hnswlib index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized hnswlib index file should not be empty"
        );

        let _ = std::fs::remove_file(&filepath);
    }

    /// Passing a filename containing an interior NUL byte must surface as an
    /// `InvalidPath` error rather than panicking inside the serializer.
    #[test]
    fn test_cagra_serialize_rejects_interior_nul() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::try_new().unwrap();
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let index =
            Index::build(&res, &build_params, &*dataset).expect("failed to build cagra index");

        // `PathBuf::from` on Unix preserves arbitrary bytes, so we can embed a
        // NUL byte in the path and confirm the helper rejects it.
        let bad_path = std::path::PathBuf::from("/tmp/has\0nul.bin");
        let err = index
            .serialize(&res, &bad_path, true)
            .expect_err("serialize should reject paths with interior NUL");
        assert!(matches!(err, CagraError::InvalidPath(_)), "expected InvalidPath, got {err:?}");
    }
}
