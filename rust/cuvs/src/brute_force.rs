/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Brute-force (exact) k-NN.
//!
//! Build an [`Index`] over a dataset, then [`search`](Index::search) it with
//! device-resident queries and output buffers. Tensors are borrowed through the
//! [`AsDlTensor`] /
//! [`AsDlTensorMut`] traits; see the
//! [`dlpack`](crate::dlpack) module for the tensor model and `examples/cagra.rs`
//! for the same build/search workflow.

use std::ffi::CString;
use std::io::{Write, stderr};
use std::marker::PhantomData;
use std::path::Path;

use crate::distance_type::DistanceType;
use crate::dlpack::{AsDlTensor, AsDlTensorMut};
use crate::error::{Error, Result, check_cuvs};
use crate::resources::Resources;

/// Brute Force KNN Index
#[derive(Debug)]
pub struct Index<'d> {
    inner: ffi::cuvsBruteForceIndex_t,
    // cuVS brute_force::index stores a non-owning view into the dataset.
    // Keep the Rust borrow alive for as long as the C++ index may read it.
    _dataset: PhantomData<&'d ()>,
}

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

impl<'d> Index<'d> {
    /// Builds a brute-force index over `dataset` for exact k-NN search.
    ///
    /// `metric` selects the distance and `metric_arg` is the optional `p` for
    /// Minkowski distances (defaults to 2). `dataset` is a row-major matrix on
    /// the host or device implementing [`AsDlTensor`]; the
    /// C++ index keeps a non-owning view of it, so the returned [`Index`] borrows
    /// it for `'d` and cannot outlive it.
    pub fn build<T>(
        res: &Resources,
        metric: DistanceType,
        metric_arg: Option<f32>,
        dataset: &'d T,
    ) -> Result<Index<'d>>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsBruteForceBuild(
                res.0,
                dataset.to_c().as_mut_ptr(),
                metric,
                metric_arg.unwrap_or(2.0),
                index.inner,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index<'d>> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsBruteForceIndex_t>::uninit();
            check_cuvs(ffi::cuvsBruteForceIndexCreate(index.as_mut_ptr()))?;
            Ok(Index { inner: index.assume_init(), _dataset: PhantomData })
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

            check_cuvs(ffi::cuvsBruteForceSearch(
                res.0,
                self.inner,
                queries.to_c().as_mut_ptr(),
                neighbors.to_c().as_mut_ptr(),
                distances.to_c().as_mut_ptr(),
                prefilter,
            ))
        }
    }

    /// Save the Brute Force index to file.
    ///
    /// The serialization format can be subject to change, therefore loading an
    /// index saved with a previous version of cuVS is not guaranteed to work.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    pub fn serialize<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        unsafe { check_cuvs(ffi::cuvsBruteForceSerialize(res.0, c_filename.as_ptr(), self.inner)) }
    }

    /// Load a Brute Force index from file.
    ///
    /// The serialization format can be subject to change, therefore loading an
    /// index saved with a previous version of cuVS is not guaranteed to work.
    ///
    /// A deserialized index owns its data internally and borrows nothing from
    /// Rust, hence the `'static` dataset lifetime.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The path of the file that stores the index
    pub fn deserialize<P: AsRef<Path>>(res: &Resources, filename: P) -> Result<Index<'static>> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        // Create the Index handle first so that any error path below still runs
        // its `Drop` and releases the C-side index allocation.
        let index: Index<'static> = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsBruteForceDeserialize(res.0, c_filename.as_ptr(), index.inner))?;
        }
        Ok(index)
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

        // Create a new random dataset to index
        let n_datapoints = 16;
        let n_features = 8;
        let dataset_host = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset = DeviceTensor::from_host(&res, &dataset_host).unwrap();

        println!("dataset {:#?}", dataset_host);

        // build the brute force index
        let index =
            Index::build(&res, metric, None, &dataset).expect("failed to create brute force index");

        res.sync_stream().unwrap();

        // use the first 4 points from the dataset as queries : will test that we get them back
        // as their own nearest neighbor
        let n_queries = 4;
        let queries = dataset_host.slice(s![0..n_queries, ..]).to_owned();

        let k = 4;

        println!("queries! {:#?}", queries);
        let queries = DeviceTensor::from_host(&res, &queries).unwrap();
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();

        index.search(&res, &queries, &mut neighbors, &mut distances).unwrap();

        // Copy back to host memory
        distances.copy_to_host(&res, &mut distances_host).unwrap();
        neighbors.copy_to_host(&res, &mut neighbors_host).unwrap();
        res.sync_stream().unwrap();

        println!("distances {:#?}", distances_host);
        println!("neighbors {:#?}", neighbors_host);

        // nearest neighbors should be themselves, since queries are from the
        // dataset
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    /*
        #[test]
        fn test_cosine() {
            test_bfknn(DistanceType::CosineExpanded);
        }
    */

    #[test]
    fn test_l2() {
        test_bfknn(DistanceType::L2Expanded);
    }

    const N_DATAPOINTS: usize = 16;
    const N_FEATURES: usize = 8;

    /// Search the first `n_queries` rows of `dataset` against `index` and assert
    /// each query finds itself as the top-1 neighbor.
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

        index.search(res, &queries, &mut neighbors, &mut distances).expect("search failed");

        distances.copy_to_host(res, &mut distances_host).unwrap();
        neighbors.copy_to_host(res, &mut neighbors_host).unwrap();
        res.sync_stream().unwrap();

        for i in 0..n_queries {
            assert_eq!(
                neighbors_host[[i, 0]],
                i as i64,
                "query {i} should be its own nearest neighbor"
            );
        }
    }

    #[test]
    fn test_brute_force_serialize_deserialize() {
        let res = Resources::new().unwrap();

        // Keep `device_dataset` alive for the whole test: the C++ index stores a
        // non-owning view into it, so it must not be dropped while `index` lives.
        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let device_dataset = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, DistanceType::L2Expanded, None, &device_dataset)
            .expect("failed to build brute force index");
        res.sync_stream().unwrap();

        let unique = format!(
            "test_brute_force_index_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        );
        let filepath = std::env::temp_dir().join(unique);
        index.serialize(&res, &filepath).expect("failed to serialize brute force index");

        assert!(filepath.exists(), "serialized index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized index file should not be empty"
        );

        let loaded_index =
            Index::deserialize(&res, &filepath).expect("failed to deserialize brute force index");

        // The deserialized index should still find each query as its own
        // nearest neighbor.
        search_and_verify_self_neighbors(&res, &loaded_index, &dataset, 4, 4);

        let _ = std::fs::remove_file(&filepath);
    }

    /// Passing a filename containing an interior NUL byte must surface as an
    /// `InvalidArgument` error rather than panicking inside the serializer.
    #[test]
    fn test_brute_force_serialize_rejects_interior_nul() {
        let res = Resources::new().unwrap();

        let dataset = ndarray::Array::<f32, _>::random(
            (N_DATAPOINTS, N_FEATURES),
            Uniform::new(0., 1.0).unwrap(),
        );
        let device_dataset = DeviceTensor::from_host(&res, &dataset).unwrap();
        let index = Index::build(&res, DistanceType::L2Expanded, None, &device_dataset)
            .expect("failed to build brute force index");
        res.sync_stream().unwrap();

        // `PathBuf::from` on Unix preserves arbitrary bytes, so we can embed a
        // NUL byte in the path and confirm the helper rejects it.
        let bad_path = std::path::PathBuf::from("/tmp/has\0nul.bin");
        let err = index
            .serialize(&res, &bad_path)
            .expect_err("serialize should reject paths with interior NUL");
        assert!(matches!(err, Error::InvalidArgument(_)), "expected InvalidArgument, got {err:?}");
    }
}
