/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar quantizer.
//!
//! The scalar quantizer performs a linear mapping of an interval of the input
//! float range onto the full range of an 8-bit integer. The interval is
//! derived during [`Quantizer::train`] from the dataset, optionally clipping a
//! configurable fraction of outliers (see
//! [`ScalarQuantizerParams::set_quantile`]).

use std::fmt;
use std::io::{Write, stderr};

use crate::dlpack::{AsDlTensor, AsDlTensorMut, DLTensorView};
use crate::error::{Error, Result, check_cuvs};
use crate::resources::Resources;

/// The C API reinterprets `i8` buffers without validating dtype; guard
/// Rust-side so a wrong-dtype tensor surfaces as `InvalidArgument` instead
/// of memory corruption.
fn expect_i8_view(view: &DLTensorView, arg: &str) -> Result<()> {
    let dtype = view.dtype();
    if dtype.code != ffi::DLDataTypeCode::kDLInt as u8 || dtype.bits != 8 || dtype.lanes != 1 {
        return Err(Error::InvalidArgument(format!(
            "{arg} must be an i8 tensor (got code={}, bits={}, lanes={})",
            dtype.code, dtype.bits, dtype.lanes
        )));
    }
    Ok(())
}

/// Parameters controlling how a [`Quantizer`] is trained.
pub struct ScalarQuantizerParams(pub ffi::cuvsScalarQuantizerParams_t);

impl ScalarQuantizerParams {
    /// Returns a new `ScalarQuantizerParams` populated with default values.
    pub fn new() -> Result<ScalarQuantizerParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsScalarQuantizerParams_t>::uninit();
            check_cuvs(ffi::cuvsScalarQuantizerParamsCreate(params.as_mut_ptr()))?;
            Ok(ScalarQuantizerParams(params.assume_init()))
        }
    }

    /// Sets the fraction of the data that is kept once outliers at the top and
    /// bottom of the distribution have been ignored.
    ///
    /// Must be within the range `(0, 1]`. The default is `0.99`.
    pub fn set_quantile(self, quantile: f32) -> ScalarQuantizerParams {
        unsafe {
            (*self.0).quantile = quantile;
        }
        self
    }
}

impl fmt::Debug for ScalarQuantizerParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug impl: the default would just print the raw pointer
        write!(f, "ScalarQuantizerParams({:?})", unsafe { *self.0 })
    }
}

impl Drop for ScalarQuantizerParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsScalarQuantizerParamsDestroy(self.0) }) {
            let _ = write!(stderr(), "failed to call cuvsScalarQuantizerParamsDestroy {:?}", e);
        }
    }
}

/// A trained scalar quantizer.
///
/// Build one with [`Quantizer::train`], then use [`Quantizer::transform`] to
/// quantize a float dataset into int8 and [`Quantizer::inverse_transform`] to
/// reconstruct an approximation of the original float values.
#[derive(Debug)]
pub struct Quantizer(ffi::cuvsScalarQuantizer_t);

impl Quantizer {
    /// Creates a new, untrained quantizer.
    fn new() -> Result<Quantizer> {
        unsafe {
            let mut quantizer = std::mem::MaybeUninit::<ffi::cuvsScalarQuantizer_t>::uninit();
            check_cuvs(ffi::cuvsScalarQuantizerCreate(quantizer.as_mut_ptr()))?;
            Ok(Quantizer(quantizer.assume_init()))
        }
    }

    /// Trains a scalar quantizer on `dataset` for later use in quantizing data.
    ///
    /// `dataset` is a row-major `f32`, `f16`, or `f64` matrix on the host or
    /// device implementing [`AsDlTensor`].
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters controlling the quantization (e.g. quantile)
    /// * `dataset` - The training dataset
    pub fn train<T>(
        res: &Resources,
        params: &ScalarQuantizerParams,
        dataset: &T,
    ) -> Result<Quantizer>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let quantizer = Quantizer::new()?;
        unsafe {
            check_cuvs(ffi::cuvsScalarQuantizerTrain(
                res.0,
                params.0,
                dataset.to_c().as_mut_ptr(),
                quantizer.0,
            ))?;
        }
        Ok(quantizer)
    }

    /// Quantizes `dataset` into `out`.
    ///
    /// `dataset` and `out` implement [`AsDlTensor`] /
    /// [`AsDlTensorMut`]; `out` is written in place.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `dataset` - A row-major `f32`, `f16`, or `f64` matrix to quantize, shape `(m, n)`
    /// * `out` - A row-major `i8` matrix that receives the quantized data, shape `(m, n)`
    ///   — the output dtype must be `i8`: the C API does not validate it and will
    ///   reinterpret the buffer otherwise (unlike `inverse_transform`, whose output
    ///   dtype is validated)
    pub fn transform<D, O>(&self, res: &Resources, dataset: &D, out: &mut O) -> Result<()>
    where
        D: AsDlTensor + ?Sized,
        O: AsDlTensorMut + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let out = out.as_dl_tensor_mut()?;
        expect_i8_view(&out, "transform output")?;
        unsafe {
            check_cuvs(ffi::cuvsScalarQuantizerTransform(
                res.0,
                self.0,
                dataset.to_c().as_mut_ptr(),
                out.to_c().as_mut_ptr(),
            ))
        }
    }

    /// Reconstructs an approximation of the original float dataset from
    /// previously quantized data.
    ///
    /// Note that scalar quantization is lossy, so the reconstructed values only
    /// approximate the originals. `dataset` and `out` implement
    /// [`AsDlTensor`] / [`AsDlTensorMut`];
    /// `out` is written in place.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `dataset` - A row-major `i8` matrix of quantized data, shape `(m, n)`
    /// * `out` - A row-major `f32` matrix that receives the reconstructed data, shape `(m, n)`
    pub fn inverse_transform<D, O>(&self, res: &Resources, dataset: &D, out: &mut O) -> Result<()>
    where
        D: AsDlTensor + ?Sized,
        O: AsDlTensorMut + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        expect_i8_view(&dataset, "inverse_transform input")?;
        let out = out.as_dl_tensor_mut()?;
        unsafe {
            check_cuvs(ffi::cuvsScalarQuantizerInverseTransform(
                res.0,
                self.0,
                dataset.to_c().as_mut_ptr(),
                out.to_c().as_mut_ptr(),
            ))
        }
    }
}

impl Drop for Quantizer {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsScalarQuantizerDestroy(self.0) }) {
            let _ = write!(stderr(), "failed to call cuvsScalarQuantizerDestroy {:?}", e);
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
    fn test_scalar_quantizer_params() {
        let params = ScalarQuantizerParams::new().unwrap().set_quantile(0.95);

        // make sure the setter actually updated the internal c-struct
        unsafe {
            assert_eq!((*params.0).quantile, 0.95);
        }
    }

    #[test]
    fn test_scalar_quantizer_roundtrip() {
        let res = Resources::new().unwrap();

        // Create a random dataset to quantize. The data range is [0, 10), so
        // the int8 quantization step is roughly 10 / 256 ~= 0.04.
        let n_rows = 1024;
        let n_cols = 16;
        let data_lo = 0.0f32;
        let data_hi = 10.0f32;
        let dataset = ndarray::Array::<f32, _>::random(
            (n_rows, n_cols),
            Uniform::new(data_lo, data_hi).unwrap(),
        );
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        // Train the quantizer (use the full range so we don't clip outliers).
        let params = ScalarQuantizerParams::new().unwrap().set_quantile(1.0);
        let quantizer = Quantizer::train(&res, &params, &dataset_device).unwrap();

        // Quantize the dataset into int8.
        let mut quantized_host = ndarray::Array::<i8, _>::zeros((n_rows, n_cols));
        let mut quantized = DeviceTensor::<i8>::zeros(&res, &[n_rows, n_cols]).unwrap();
        quantizer.transform(&res, &dataset_device, &mut quantized).unwrap();
        quantized.copy_to_host(&res, &mut quantized_host).unwrap();

        // The quantized values should span a good chunk of the int8 range,
        // confirming the transform actually did something.
        let q_min = *quantized_host.iter().min().unwrap();
        let q_max = *quantized_host.iter().max().unwrap();
        assert!(
            q_max as i32 - q_min as i32 > 200,
            "quantized values should span most of the int8 range, got [{q_min}, {q_max}]"
        );

        // Reconstruct an approximation of the original f32 values.
        let mut reconstructed_host = ndarray::Array::<f32, _>::zeros((n_rows, n_cols));
        let mut reconstructed = DeviceTensor::<f32>::zeros(&res, &[n_rows, n_cols]).unwrap();
        quantizer.inverse_transform(&res, &quantized, &mut reconstructed).unwrap();
        reconstructed.copy_to_host(&res, &mut reconstructed_host).unwrap();

        // Compute the max absolute reconstruction error. It should be bounded
        // by a few quantization steps and far below the data range.
        let mut max_abs_err = 0.0f32;
        for (orig, recon) in dataset.iter().zip(reconstructed_host.iter()) {
            let err = (orig - recon).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
        }

        let data_range = data_hi - data_lo;
        // A loose epsilon: a handful of quantization steps. One step is
        // data_range / 256 ~= 0.04; allow up to ~5 steps of slack.
        let epsilon = data_range / 50.0;
        assert!(
            max_abs_err < epsilon,
            "max abs reconstruction error {max_abs_err} should be below {epsilon}"
        );
        assert!(
            max_abs_err < data_range * 0.05,
            "max abs reconstruction error {max_abs_err} should be far below data range {data_range}"
        );
    }

    #[test]
    fn test_train_unsupported_dtype_errors() {
        let res = Resources::new().unwrap();

        // The C API only supports float (16/32/64-bit) training datasets, and
        // surfaces an integer dataset as an error rather than silently
        // succeeding. (Note: a freshly created, untrained quantizer has
        // min_ == max_ == 0, which produces degenerate output but is *not*
        // reported as an error by the C API, so we exercise the dtype guard
        // instead to cover the error path.)
        let n_rows = 8;
        let n_cols = 4;
        let dataset = ndarray::Array::<i32, _>::zeros((n_rows, n_cols));
        let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();

        let params = ScalarQuantizerParams::new().unwrap();
        let result = Quantizer::train(&res, &params, &dataset_device);
        assert!(
            result.is_err(),
            "training on an unsupported (integer) dtype should return an error"
        );
    }

    #[test]
    fn test_transform_rejects_non_i8_output() {
        let res = Resources::new().unwrap();
        let n_rows = 8;
        let n_cols = 4;

        let dataset = ndarray::Array::<f32, _>::zeros((n_rows, n_cols));
        let mut dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
        let params = ScalarQuantizerParams::new().unwrap();
        let quantizer = Quantizer::train(&res, &params, &dataset_device).unwrap();

        // The C API would silently reinterpret a non-i8 output buffer;
        // the wrapper must reject it before any FFI happens.
        let bad_out = ndarray::Array::<f32, _>::zeros((n_rows, n_cols));
        let mut bad_out_device = DeviceTensor::from_host(&res, &bad_out).unwrap();
        let result = quantizer.transform(&res, &dataset_device, &mut bad_out_device);
        assert!(
            matches!(
                &result,
                Err(Error::InvalidArgument(msg))
                    if msg.contains("transform output") && msg.contains("i8 tensor")
            ),
            "transform must reject a non-i8 output tensor via the dtype guard, got {result:?}"
        );

        // Same guard on the inverse path's input.
        let result = quantizer.inverse_transform(&res, &bad_out_device, &mut dataset_device);
        assert!(
            matches!(
                &result,
                Err(Error::InvalidArgument(msg))
                    if msg.contains("inverse_transform input") && msg.contains("i8 tensor")
            ),
            "inverse_transform must reject a non-i8 input tensor via the dtype guard, got {result:?}"
        );
    }
}
