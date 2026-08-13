/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Test-only tensor adapters.
//!
//! [`DeviceTensor`] is an RMM-backed device matrix, and the `ndarray` host
//! adapters below implement [`AsDlTensor`]/[`AsDlTensorMut`] for plain host
//! arrays. We use `ndarray` only as a dev-dependency to assist with unit tests.

use std::marker::PhantomData;

use crate::dlpack::{AsDlTensor, AsDlTensorMut, DLPackError, DLTensorView, DLTensorViewMut, DType};
use crate::error::check_cuvs;
use crate::ffi;
use crate::resources::Resources;

// Test helpers can fail with either a `LibraryError` or a `DLPackError`; a boxed
// error keeps the (test-only) surface simple.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) struct DeviceTensor<'res, T: DType> {
    data: *mut std::ffi::c_void,
    shape: Vec<i64>,
    capacity_bytes: usize,
    resources: &'res Resources,
    _marker: PhantomData<T>,
}

impl<'res, T: DType> DeviceTensor<'res, T> {
    pub(crate) fn zeros(res: &'res Resources, shape: &[usize]) -> Result<Self> {
        let capacity_bytes = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            check_cuvs(ffi::cuvsRMMAlloc(res.handle(), &mut data, capacity_bytes))?;
        }

        Ok(Self {
            data,
            shape: shape.iter().map(|&dim| dim as i64).collect(),
            capacity_bytes,
            resources: res,
            _marker: PhantomData,
        })
    }

    pub(crate) fn from_host<D>(res: &'res Resources, host: &ndarray::ArrayRef<T, D>) -> Result<Self>
    where
        D: ndarray::Dimension,
    {
        let shape: Vec<usize> = host.shape().to_vec();
        let mut device = Self::zeros(res, &shape)?;

        let host_shape: Vec<i64> = host.shape().iter().map(|&dim| dim as i64).collect();
        let host_strides: Option<Vec<i64>> = if host.is_standard_layout() {
            None
        } else {
            Some(host.strides().iter().map(|&stride| stride as i64).collect())
        };
        let host = unsafe {
            DLTensorView::from_raw_parts(
                host.as_ptr() as *mut _,
                ffi::DLDevice { device_type: ffi::DLDeviceType::kDLCPU, device_id: 0 },
                &host_shape,
                host_strides.as_deref(),
                T::dl_dtype(),
            )?
        };
        let device_view = device.as_dl_tensor_mut()?;
        unsafe {
            check_cuvs(ffi::cuvsMatrixCopy(
                res.handle(),
                host.to_c().as_mut_ptr(),
                device_view.to_c().as_mut_ptr(),
            ))?;
        }

        Ok(device)
    }

    pub(crate) fn copy_to_host<D>(
        &self,
        res: &Resources,
        host: &mut ndarray::ArrayRef<T, D>,
    ) -> Result<()>
    where
        D: ndarray::Dimension,
    {
        let host_shape: Vec<i64> = host.shape().iter().map(|&dim| dim as i64).collect();
        let host_strides: Option<Vec<i64>> = if host.is_standard_layout() {
            None
        } else {
            Some(host.strides().iter().map(|&stride| stride as i64).collect())
        };
        let host = unsafe {
            DLTensorViewMut::from_raw_parts(
                host.as_mut_ptr() as *mut _,
                ffi::DLDevice { device_type: ffi::DLDeviceType::kDLCPU, device_id: 0 },
                &host_shape,
                host_strides.as_deref(),
                T::dl_dtype(),
            )?
        };
        let device = self.as_dl_tensor()?;
        unsafe {
            check_cuvs(ffi::cuvsMatrixCopy(
                res.handle(),
                device.to_c().as_mut_ptr(),
                host.to_c().as_mut_ptr(),
            ))?;
        }

        Ok(())
    }
}

impl<T: DType> Drop for DeviceTensor<'_, T> {
    fn drop(&mut self) {
        if !self.data.is_null() {
            let _ = unsafe {
                ffi::cuvsRMMFree(self.resources.handle(), self.data, self.capacity_bytes)
            };
        }
    }
}

impl<T: DType> AsDlTensor for DeviceTensor<'_, T> {
    fn as_dl_tensor(&self) -> std::result::Result<DLTensorView<'_>, DLPackError> {
        unsafe {
            DLTensorView::from_raw_parts(
                self.data,
                ffi::DLDevice { device_type: ffi::DLDeviceType::kDLCUDA, device_id: 0 },
                &self.shape,
                None,
                T::dl_dtype(),
            )
        }
    }
}

impl<T: DType> AsDlTensorMut for DeviceTensor<'_, T> {
    fn as_dl_tensor_mut(&mut self) -> std::result::Result<DLTensorViewMut<'_>, DLPackError> {
        unsafe {
            DLTensorViewMut::from_raw_parts(
                self.data,
                ffi::DLDevice { device_type: ffi::DLDeviceType::kDLCUDA, device_id: 0 },
                &self.shape,
                None,
                T::dl_dtype(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// ndarray host adapters (test-only)
// ---------------------------------------------------------------------------

fn array_layout<A, D>(arr: &ndarray::ArrayRef<A, D>) -> (Vec<i64>, Option<Vec<i64>>)
where
    D: ndarray::Dimension,
{
    let shape = arr.shape().iter().map(|&d| d as i64).collect();
    let strides = if arr.is_standard_layout() {
        None
    } else {
        Some(arr.strides().iter().map(|&s| s as i64).collect())
    };
    (shape, strides)
}

impl<A, D> AsDlTensor for ndarray::ArrayRef<A, D>
where
    A: DType,
    D: ndarray::Dimension,
{
    fn as_dl_tensor(&self) -> std::result::Result<DLTensorView<'_>, DLPackError> {
        let (shape, strides) = array_layout(self);
        unsafe {
            DLTensorView::from_raw_parts(
                self.as_ptr() as *mut _,
                ffi::DLDevice { device_type: ffi::DLDeviceType::kDLCPU, device_id: 0 },
                &shape,
                strides.as_deref(),
                A::dl_dtype(),
            )
        }
    }
}

impl<A, D> AsDlTensorMut for ndarray::ArrayRef<A, D>
where
    A: DType,
    D: ndarray::Dimension,
{
    fn as_dl_tensor_mut(&mut self) -> std::result::Result<DLTensorViewMut<'_>, DLPackError> {
        let (shape, strides) = array_layout(self);
        unsafe {
            DLTensorViewMut::from_raw_parts(
                self.as_mut_ptr() as *mut _,
                ffi::DLDevice { device_type: ffi::DLDeviceType::kDLCPU, device_id: 0 },
                &shape,
                strides.as_deref(),
                A::dl_dtype(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The ndarray adapter must report a contiguous array as contiguous (no
    // strides) with the correct shape, device, and dtype.
    #[test]
    fn ndarray_contiguous_view_omits_strides() {
        let arr = ndarray::Array2::<f32>::zeros((10, 20));
        let view = arr.as_dl_tensor().unwrap();

        assert_eq!(view.shape(), &[10, 20]);
        assert!(view.strides().is_none());
        assert_eq!(view.device().device_type, ffi::DLDeviceType::kDLCPU);
        assert_eq!(view.dtype().code, ffi::DLDataTypeCode::kDLFloat as u8);
        assert_eq!(view.dtype().bits, 32);
    }

    // A non-standard layout must surface explicit strides so cuVS reads the
    // data in the right order.
    #[test]
    fn ndarray_transposed_view_reports_strides() {
        let arr = ndarray::Array2::<f32>::zeros((10, 20));
        let transposed = arr.t();
        let view = transposed.as_dl_tensor().unwrap();

        assert_eq!(view.shape(), &[20, 10]);
        assert_eq!(view.strides(), Some(&[1i64, 20][..]));
    }
}
