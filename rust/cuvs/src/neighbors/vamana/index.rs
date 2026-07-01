/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{Write, stderr};

use super::{IndexParams, VamanaError};
use crate::dlpack::AsDlTensor;
use crate::error::check_cuvs;
use crate::resources::Resources;

type Result<T> = std::result::Result<T, VamanaError>;

/// Vamana ANN index.
#[derive(Debug)]
pub struct Index(ffi::cuvsVamanaIndex_t);

impl Index {
    /// Builds a Vamana index for efficient DiskANN search.
    ///
    /// The build uses the Vamana insertion-based algorithm: starting from an
    /// empty graph it iteratively inserts batches of nodes, performing a greedy
    /// search for each inserted vector and connecting it to all nodes traversed;
    /// reverse edges are added and `robustPrune` is applied to improve quality.
    /// [`IndexParams`] controls the degree of the final graph.
    ///
    /// `dataset` is a row-major matrix on the host or device implementing
    /// [`AsDlTensor`]; it is copied into the index, so `Index` carries no
    /// lifetime.
    pub fn build<T>(res: &Resources, params: &IndexParams, dataset: &T) -> Result<Index>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsVamanaBuild(
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
            let mut index = std::mem::MaybeUninit::<ffi::cuvsVamanaIndex_t>::uninit();
            check_cuvs(ffi::cuvsVamanaIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Saves the Vamana index to a file.
    ///
    /// Matches the on-disk format used by the DiskANN open-source repository,
    /// so the serialized index can be consumed there for graph search.
    ///
    /// `filename` is the file prefix under which the index is saved;
    /// `include_dataset` controls whether the dataset is embedded.
    pub fn serialize(self, res: &Resources, filename: &str, include_dataset: bool) -> Result<()> {
        let c_filename = CString::new(filename)?;
        check_cuvs(unsafe {
            ffi::cuvsVamanaSerialize(res.handle(), c_filename.as_ptr(), self.0, include_dataset)
        })?;
        Ok(())
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
        let build_params = IndexParams::builder().build().unwrap();
        let res = Resources::new().unwrap();

        let n_datapoints = 1024;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );

        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let _index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to create vamana index");
    }
}
