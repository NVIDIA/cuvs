/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! K-means clustering.
//!
//! [`fit`] computes cluster centroids for a dataset, [`predict`] assigns points
//! to clusters, and [`cluster_cost`] reports the inertia. All inputs and outputs
//! reside in device memory and are borrowed through the `AsDlTensor` /
//! `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack) module for the
//! tensor model.

mod params;

pub use params::Params;

use crate::dlpack::{AsDlTensor, AsDlTensorMut, DLPackError};
use crate::error::{LibraryError, check_cuvs};
use crate::resources::Resources;

type Result<T> = std::result::Result<T, KMeansError>;

/// Error type for k-means operations.
#[derive(Debug, thiserror::Error)]
pub enum KMeansError {
    /// The cuVS C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
}

/// Fits k-means centroids to `x`, returning `(inertia, n_iterations)`.
///
/// `x` (shape `m × k`) is the input matrix and `centroids` (shape
/// `n_clusters × k`) receives the fitted centroids; `sample_weight` is an
/// optional per-sample weight. All reside in device memory and implement
/// [`AsDlTensor`] / [`AsDlTensorMut`].
pub fn fit<X, W, C>(
    res: &Resources,
    params: &Params,
    x: &X,
    sample_weight: Option<&W>,
    centroids: &mut C,
) -> Result<(f64, i32)>
where
    X: AsDlTensor + ?Sized,
    W: AsDlTensor + ?Sized,
    C: AsDlTensorMut + ?Sized,
{
    let x = x.as_dl_tensor()?;
    let sample_weight = sample_weight.map(|w| w.as_dl_tensor()).transpose()?;
    let centroids = centroids.as_dl_tensor_mut()?;
    let mut inertia: f64 = 0.0;
    let mut niter: i32 = 0;
    let mut sample_weight_c = sample_weight.as_ref().map(|w| w.to_c());
    let sample_weight_ptr =
        sample_weight_c.as_mut().map(|w| w.as_mut_ptr()).unwrap_or(std::ptr::null_mut());

    unsafe {
        check_cuvs(ffi::cuvsKMeansFit(
            res.handle(),
            params.handle(),
            x.to_c().as_mut_ptr(),
            sample_weight_ptr,
            centroids.to_c().as_mut_ptr(),
            &mut inertia as *mut f64,
            &mut niter as *mut i32,
        ))?;
    }
    Ok((inertia, niter))
}

/// Assigns each row of `x` to its nearest centroid, writing cluster labels into
/// `labels` and returning the inertia.
///
/// `x` (shape `m × k`), `centroids` (shape `n_clusters × k`), the optional
/// `sample_weight`, and `labels` (shape `m × 1`) reside in device memory and
/// implement [`AsDlTensor`] / [`AsDlTensorMut`]. `normalize_weight` selects
/// whether the sample weights are normalized.
pub fn predict<X, W, C, L>(
    res: &Resources,
    params: &Params,
    x: &X,
    sample_weight: Option<&W>,
    centroids: &C,
    labels: &mut L,
    normalize_weight: bool,
) -> Result<f64>
where
    X: AsDlTensor + ?Sized,
    W: AsDlTensor + ?Sized,
    C: AsDlTensor + ?Sized,
    L: AsDlTensorMut + ?Sized,
{
    let x = x.as_dl_tensor()?;
    let sample_weight = sample_weight.map(|w| w.as_dl_tensor()).transpose()?;
    let centroids = centroids.as_dl_tensor()?;
    let labels = labels.as_dl_tensor_mut()?;
    let mut inertia: f64 = 0.0;
    let mut sample_weight_c = sample_weight.as_ref().map(|w| w.to_c());
    let sample_weight_ptr =
        sample_weight_c.as_mut().map(|w| w.as_mut_ptr()).unwrap_or(std::ptr::null_mut());

    unsafe {
        check_cuvs(ffi::cuvsKMeansPredict(
            res.handle(),
            params.handle(),
            x.to_c().as_mut_ptr(),
            sample_weight_ptr,
            centroids.to_c().as_mut_ptr(),
            labels.to_c().as_mut_ptr(),
            normalize_weight,
            &mut inertia as *mut f64,
        ))?;
    }
    Ok(inertia)
}

/// Computes the k-means cost (inertia) of `x` against existing `centroids`.
///
/// `x` (shape `m × k`) and `centroids` (shape `n_clusters × k`) reside in device
/// memory and implement [`AsDlTensor`].
pub fn cluster_cost<X, C>(res: &Resources, x: &X, centroids: &C) -> Result<f64>
where
    X: AsDlTensor + ?Sized,
    C: AsDlTensor + ?Sized,
{
    let x = x.as_dl_tensor()?;
    let centroids = centroids.as_dl_tensor()?;
    let mut inertia: f64 = 0.0;

    unsafe {
        check_cuvs(ffi::cuvsKMeansClusterCost(
            res.handle(),
            x.to_c().as_mut_ptr(),
            centroids.to_c().as_mut_ptr(),
            &mut inertia as *mut f64,
        ))?;
    }
    Ok(inertia)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::DeviceTensor;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_kmeans() {
        let res = Resources::new().unwrap();

        let n_clusters = 4;

        let n_datapoints = 256;
        let n_features = 16;
        let dataset_host = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset = DeviceTensor::from_host(&res, &dataset_host).unwrap();

        let centroids_host = ndarray::Array::<f32, _>::zeros((n_clusters, n_features));
        let mut centroids = DeviceTensor::from_host(&res, &centroids_host).unwrap();

        let params = Params::builder().n_clusters(n_clusters as i32).build().unwrap();

        let original_inertia = cluster_cost(&res, &dataset, &centroids).unwrap();

        let (inertia, n_iter) =
            fit(&res, &params, &dataset, None::<&DeviceTensor<'_, f32>>, &mut centroids).unwrap();

        assert!(inertia < original_inertia);
        assert!(n_iter >= 1);

        let mut labels_host = ndarray::Array::<i32, _>::zeros((n_clusters,));
        let mut labels = DeviceTensor::<i32>::zeros(&res, &[n_clusters]).unwrap();

        predict(
            &res,
            &params,
            &centroids,
            None::<&DeviceTensor<'_, f32>>,
            &centroids,
            &mut labels,
            false,
        )
        .unwrap();

        labels.copy_to_host(&res, &mut labels_host).unwrap();
        assert_eq!(labels_host[[0,]], 0);
        assert_eq!(labels_host[[1,]], 1);
        assert_eq!(labels_host[[2,]], 2);
        assert_eq!(labels_host[[3,]], 3);
    }
}
