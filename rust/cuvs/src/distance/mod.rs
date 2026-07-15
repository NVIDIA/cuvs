/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pairwise distance computation.
//!
//! [`pairwise_distance`] computes all pairwise distances between two device
//! matrices. Inputs and output are borrowed through the
//! [`AsDlTensor`] /
//! [`AsDlTensorMut`] traits; see the
//! [`dlpack`](crate::dlpack) module for the tensor model.

use crate::distance_type::DistanceType;
use crate::dlpack::{AsDlTensor, AsDlTensorMut};
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;

/// Computes all pairwise distances between the rows of `x` (shape `m × k`) and
/// `y` (shape `n × k`), writing the `m × n` result into `distances`.
///
/// `x`, `y`, and `distances` reside in device memory and implement
/// [`AsDlTensor`] /
/// [`AsDlTensorMut`]. `metric` selects the distance;
/// `metric_arg` is the optional `p` for Minkowski distances (defaults to 2).
pub fn pairwise_distance<X, Y, D>(
    res: &Resources,
    x: &X,
    y: &Y,
    distances: &mut D,
    metric: DistanceType,
    metric_arg: Option<f32>,
) -> Result<()>
where
    X: AsDlTensor + ?Sized,
    Y: AsDlTensor + ?Sized,
    D: AsDlTensorMut + ?Sized,
{
    let x = x.as_dl_tensor()?;
    let y = y.as_dl_tensor()?;
    let distances = distances.as_dl_tensor_mut()?;
    unsafe {
        check_cuvs(ffi::cuvsPairwiseDistance(
            res.0,
            x.to_c().as_mut_ptr(),
            y.to_c().as_mut_ptr(),
            distances.to_c().as_mut_ptr(),
            metric,
            metric_arg.unwrap_or(2.0),
        ))
    }
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

        // Create a new random dataset to index
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
            None,
        )
        .unwrap();

        // Copy back to host memory
        distances.copy_to_host(&res, &mut distances_host).unwrap();

        // Self distance should be 0
        assert_eq!(distances_host[[0, 0]], 0.0);
    }
}
