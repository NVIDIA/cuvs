/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{Write, stderr};

use crate::dlpack::AsDlTensor;
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;
use crate::vamana::IndexParams;

/// Vamana ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsVamanaIndex_t);

impl Index {
    /// Builds Vamana Index for efficient DiskANN search
    ///
    /// The build uses the Vamana insertion-based algorithm to create the graph. The algorithm
    /// starts with an empty graph and iteratively inserts batches of nodes. Each batch involves
    /// performing a greedy search for each vector to be inserted, and inserting it with edges to
    /// all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied
    /// to improve graph quality. The index_params struct controls the degree of the final graph.
    ///
    /// `dataset` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]; it is copied into the index.
    pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::new()?;
        // `cuvsVamanaBuild` copies the dataset into the index, so the index does not
        // retain a view into `dataset`; the borrow only needs to be valid for the
        // duration of this call. That is why `Index` carries no lifetime.
        unsafe {
            check_cuvs(ffi::cuvsVamanaBuild(
                res.0,
                params.0,
                dataset.to_c().as_mut_ptr(),
                index.0,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsVamanaIndex_t>::uninit();
            check_cuvs(ffi::cuvsVamanaIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Save Vamana index to file
    ///
    /// Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility.
    ///
    /// Serialized Index is to be used by the DiskANN open-source repository for graph search.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file prefix for where the index is sazved
    /// * `include_dataset` - whether to include the dataset in the serialized index
    pub fn serialize(self, res: &Resources, filename: &str, include_dataset: bool) -> Result<()> {
        let c_filename = CString::new(filename).unwrap();
        unsafe {
            check_cuvs(ffi::cuvsVamanaSerialize(
                res.0,
                c_filename.as_ptr(),
                self.0,
                include_dataset,
            ))
        }
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsVamanaIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsVamanaIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::DeviceTensor;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_vamana() {
        let build_params = IndexParams::new().unwrap();

        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 1024;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        // build the vamana index
        let _index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to create vamana index");
    }
}
