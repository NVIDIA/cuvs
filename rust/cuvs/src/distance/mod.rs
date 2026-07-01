/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Distance metrics and pairwise distance computation.
//!
//! [`DistanceType`] selects the metric used by every index and by
//! [`pairwise_distance`]. Inputs and output are borrowed through the
//! `AsDlTensor` / `AsDlTensorMut` traits; see the [`dlpack`](crate::dlpack)
//! module for the tensor model.

use crate::dlpack::{AsDlTensor, AsDlTensorMut, DLPackError};
use crate::error::{LibraryError, check_cuvs};
use crate::resources::Resources;

const DEFAULT_METRIC_ARG: f32 = 2.0;

/// Distance metric used for building and searching nearest neighbor indices.
#[derive(Debug, Copy, Clone, PartialEq)]
#[non_exhaustive]
pub enum DistanceType {
    /// L2 (squared Euclidean) distance.
    L2Expanded,
    /// L2 distance with square root.
    L2SqrtExpanded,
    /// Cosine distance.
    CosineExpanded,
    /// L1 (Manhattan) distance.
    L1,
    /// L2 distance (unexpanded form).
    L2Unexpanded,
    /// L2 distance with square root (unexpanded form).
    L2SqrtUnexpanded,
    /// Inner product.
    InnerProduct,
    /// Chebyshev (L-infinity) distance.
    Linf,
    /// Canberra distance.
    Canberra,
    /// Generalized Minkowski (Lp) distance with exponent `p`.
    LpUnexpanded(f32),
    /// Correlation distance.
    CorrelationExpanded,
    /// Jaccard distance.
    JaccardExpanded,
    /// Hellinger distance.
    HellingerExpanded,
    /// Haversine (great-circle) distance.
    Haversine,
    /// Bray-Curtis distance.
    BrayCurtis,
    /// Jensen-Shannon divergence.
    JensenShannon,
    /// Hamming distance.
    HammingUnexpanded,
    /// Kullback-Leibler divergence.
    KLDivergence,
    /// Russell-Rao distance.
    RusselRaoExpanded,
    /// Dice-Sorensen distance.
    DiceExpanded,
    /// Bitwise Hamming distance.
    BitwiseHamming,
    /// Precomputed distance matrix.
    Precomputed,
}

impl DistanceType {
    /// The `metric_arg` the C API expects: the exponent `p` for Minkowski
    /// ([`LpUnexpanded`](DistanceType::LpUnexpanded)), otherwise the default.
    pub(crate) fn metric_arg(self) -> f32 {
        match self {
            Self::LpUnexpanded(p) => p,
            _ => DEFAULT_METRIC_ARG,
        }
    }
}

impl From<DistanceType> for ffi::cuvsDistanceType {
    fn from(v: DistanceType) -> Self {
        use DistanceType::*;
        match v {
            L2Expanded => Self::L2Expanded,
            L2SqrtExpanded => Self::L2SqrtExpanded,
            CosineExpanded => Self::CosineExpanded,
            L1 => Self::L1,
            L2Unexpanded => Self::L2Unexpanded,
            L2SqrtUnexpanded => Self::L2SqrtUnexpanded,
            InnerProduct => Self::InnerProduct,
            Linf => Self::Linf,
            Canberra => Self::Canberra,
            LpUnexpanded(_) => Self::LpUnexpanded,
            CorrelationExpanded => Self::CorrelationExpanded,
            JaccardExpanded => Self::JaccardExpanded,
            HellingerExpanded => Self::HellingerExpanded,
            Haversine => Self::Haversine,
            BrayCurtis => Self::BrayCurtis,
            JensenShannon => Self::JensenShannon,
            HammingUnexpanded => Self::HammingUnexpanded,
            KLDivergence => Self::KLDivergence,
            RusselRaoExpanded => Self::RusselRaoExpanded,
            DiceExpanded => Self::DiceExpanded,
            BitwiseHamming => Self::BitwiseHamming,
            Precomputed => Self::Precomputed,
        }
    }
}

impl From<ffi::cuvsDistanceType> for DistanceType {
    fn from(v: ffi::cuvsDistanceType) -> Self {
        use ffi::cuvsDistanceType::*;
        match v {
            L2Expanded => Self::L2Expanded,
            L2SqrtExpanded => Self::L2SqrtExpanded,
            CosineExpanded => Self::CosineExpanded,
            L1 => Self::L1,
            L2Unexpanded => Self::L2Unexpanded,
            L2SqrtUnexpanded => Self::L2SqrtUnexpanded,
            InnerProduct => Self::InnerProduct,
            Linf => Self::Linf,
            Canberra => Self::Canberra,
            LpUnexpanded => Self::LpUnexpanded(DEFAULT_METRIC_ARG),
            CorrelationExpanded => Self::CorrelationExpanded,
            JaccardExpanded => Self::JaccardExpanded,
            HellingerExpanded => Self::HellingerExpanded,
            Haversine => Self::Haversine,
            BrayCurtis => Self::BrayCurtis,
            JensenShannon => Self::JensenShannon,
            HammingUnexpanded => Self::HammingUnexpanded,
            KLDivergence => Self::KLDivergence,
            RusselRaoExpanded => Self::RusselRaoExpanded,
            DiceExpanded => Self::DiceExpanded,
            BitwiseHamming => Self::BitwiseHamming,
            Precomputed => Self::Precomputed,
        }
    }
}

/// Error type for pairwise distance operations.
#[derive(Debug, thiserror::Error)]
pub enum DistanceError {
    /// The cuVS C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// Tensor conversion into DLPack metadata failed.
    #[error(transparent)]
    DLPack(#[from] DLPackError),
}

/// Computes all pairwise distances between the rows of `x` (shape `m × k`) and
/// `y` (shape `n × k`), writing the `m × n` result into `distances`.
///
/// `x`, `y`, and `distances` reside in device memory and implement
/// [`AsDlTensor`] / [`AsDlTensorMut`]. `metric` selects the distance; use
/// [`DistanceType::LpUnexpanded`] to supply the Minkowski exponent `p` (all
/// other metrics use the C API default).
pub fn pairwise_distance<X, Y, D>(
    res: &Resources,
    x: &X,
    y: &Y,
    distances: &mut D,
    metric: DistanceType,
) -> Result<(), DistanceError>
where
    X: AsDlTensor + ?Sized,
    Y: AsDlTensor + ?Sized,
    D: AsDlTensorMut + ?Sized,
{
    let x = x.as_dl_tensor()?;
    let y = y.as_dl_tensor()?;
    let distances = distances.as_dl_tensor_mut()?;
    check_cuvs(unsafe {
        ffi::cuvsPairwiseDistance(
            res.handle(),
            x.to_c().as_mut_ptr(),
            y.to_c().as_mut_ptr(),
            distances.to_c().as_mut_ptr(),
            metric.into(),
            metric.metric_arg(),
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::DeviceTensor;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_pairwise_distance() {
        let res = Resources::new().unwrap();

        let n_datapoints = 256;
        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_datapoints, n_datapoints));
        let mut distances =
            DeviceTensor::<f32>::zeros(&res, &[n_datapoints, n_datapoints]).unwrap();

        pairwise_distance(
            &res,
            &dataset_device,
            &dataset_device,
            &mut distances,
            DistanceType::L2Expanded,
        )
        .unwrap();

        distances.copy_to_host(&res, &mut distances_host).unwrap();

        // Self distance should be 0.
        assert_eq!(distances_host[[0, 0]], 0.0);
    }
}
