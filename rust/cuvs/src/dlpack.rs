/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! DLPack tensor interop.
//!
//! cuVS exchanges tensors with the C library through the DLPack ABI. This crate
//! never owns tensor storage: every entry point borrows a value that exposes a
//! view through the [`AsDlTensor`] / [`AsDlTensorMut`] traits:
//!
//! * [`DLTensorView`] — a read-only view, for inputs the C API only reads
//!   (datasets, queries).
//! * [`DLTensorViewMut`] — a writable view, for outputs the C API writes
//!   (neighbors, distances).
//!
//! A view is non-owning. The traits take `&self` / `&mut self`, so the compiler
//! ties the view's lifetime to the borrow of the value that owns the underlying
//! buffer. The view materializes a stack-local [`DLManagedTensor`] only for the
//! duration of each FFI call.
//!
//! The crate ships no public tensor type. To hand your own GPU (or host) buffer
//! to cuVS, implement [`AsDlTensor`] / [`AsDlTensorMut`] for it on top of
//! [`DLTensorView::from_raw_parts`]. Most algorithms require search inputs and
//! outputs to live in device memory. See `examples/cagra.rs` for a complete,
//! runnable adapter built on the raw CUDA runtime.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

use tinyvec::TinyVec;

pub use ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor};

/// Number of dimensions kept inline before [`TensorDims`] spills to the heap.
/// 1-D and 2-D tensors are the norm; IVF-PQ codebooks need 3-D.
const INLINE_DIMS: usize = 3;

pub(crate) type TensorDims = TinyVec<[i64; INLINE_DIMS]>;

/// Borrows a tensor as a read-only [`DLTensorView`] for tensor inputs.
///
/// Implement this for your own tensor type by calling
/// [`DLTensorView::from_raw_parts`] inside a small `unsafe` block, upholding its
/// safety contract.
///
/// # Examples
///
/// A minimal adapter for a row-major matrix in device memory:
///
/// ```
/// use cuvs::dlpack::{AsDlTensor, DLDevice, DLDeviceType, DLPackError, DLTensorView, DType};
///
/// struct GpuMatrix<T> {
///     ptr: *mut T,
///     rows: usize,
///     cols: usize,
/// }
///
/// impl<T: DType> AsDlTensor for GpuMatrix<T> {
///     fn as_dl_tensor(&self) -> Result<DLTensorView<'_>, DLPackError> {
///         let shape = [self.rows as i64, self.cols as i64];
///         // SAFETY: `ptr` points to `rows * cols` initialized elements of `T`
///         // in device 0's memory, valid while `self` is borrowed, and is
///         // row-major contiguous.
///         unsafe {
///             DLTensorView::from_raw_parts(
///                 self.ptr.cast(),
///                 DLDevice { device_type: DLDeviceType::kDLCUDA, device_id: 0 },
///                 &shape,
///                 None,
///                 T::dl_dtype(),
///             )
///         }
///     }
/// }
/// ```
pub trait AsDlTensor {
    fn as_dl_tensor(&self) -> std::result::Result<DLTensorView<'_>, DLPackError>;
}

/// Borrows a tensor as a writable [`DLTensorViewMut`] for tensor outputs.
///
/// In addition to the [`DLTensorView::from_raw_parts`] invariants, writable
/// adapters must guarantee exclusive access to the data region. The `&mut self`
/// receiver makes the compiler enforce that exclusivity for the borrow.
pub trait AsDlTensorMut {
    fn as_dl_tensor_mut(&mut self) -> std::result::Result<DLTensorViewMut<'_>, DLPackError>;
}

/// Maps a Rust element type to a DLPack [`DLDataType`].
pub trait DType {
    fn dl_dtype() -> ffi::DLDataType;
}

macro_rules! impl_dtype {
    ($ty:ty, $code:expr, $bits:expr) => {
        impl DType for $ty {
            fn dl_dtype() -> ffi::DLDataType {
                ffi::DLDataType { code: $code as u8, bits: $bits, lanes: 1 }
            }
        }
    };
}

impl_dtype!(f32, ffi::DLDataTypeCode::kDLFloat, 32);
impl_dtype!(f64, ffi::DLDataTypeCode::kDLFloat, 64);
impl_dtype!(i32, ffi::DLDataTypeCode::kDLInt, 32);
impl_dtype!(i64, ffi::DLDataTypeCode::kDLInt, 64);
impl_dtype!(u32, ffi::DLDataTypeCode::kDLUInt, 32);
impl_dtype!(u64, ffi::DLDataTypeCode::kDLUInt, 64);
impl_dtype!(u8, ffi::DLDataTypeCode::kDLUInt, 8);
impl_dtype!(i8, ffi::DLDataTypeCode::kDLInt, 8);
impl_dtype!(u16, ffi::DLDataTypeCode::kDLUInt, 16);
impl_dtype!(i16, ffi::DLDataTypeCode::kDLInt, 16);

/// Error when converting an external tensor to a DLPack view.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum DLPackError {
    /// The tensor resides on a device not supported by cuVS.
    #[error("unsupported tensor device: {0}")]
    UnsupportedDevice(String),
    /// The tensor dtype is not supported by the current adapter.
    #[error("unsupported tensor dtype: {0}")]
    UnsupportedDType(String),
    /// A strides slice did not match the tensor rank.
    #[error("strides length {strides} does not match tensor rank {ndim}")]
    StridesLenMismatch { ndim: usize, strides: usize },
    /// The source tensor reported invalid DLPack metadata.
    #[error("invalid DLPack metadata: {0}")]
    InvalidMetadata(&'static str),
}

/// A wrapper around [`ffi::DLManagedTensor`] with a lifetime attached.
pub(crate) struct ManagedTensorRef<'a> {
    pub(crate) inner: ffi::DLManagedTensor,
    _borrow: PhantomData<&'a ()>,
}

impl ManagedTensorRef<'_> {
    /// Returns a pointer to the inner [`ffi::DLManagedTensor`] for an FFI call.
    ///
    /// The pointer is valid only while this `ManagedTensorRef` is alive. This
    /// covers calling `view.to_c().as_mut_ptr()` directly as an FFI argument,
    /// because the temporary would live to the end of that statement.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::DLManagedTensor {
        &mut self.inner
    }
}

/// A non-owning, read-only DLPack tensor view.
#[must_use]
pub struct DLTensorView<'a> {
    data: *mut std::ffi::c_void,
    device: ffi::DLDevice,
    dtype: ffi::DLDataType,
    shape: TensorDims,
    strides: Option<TensorDims>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> DLTensorView<'a> {
    /// Construct a DLPack view from raw tensor metadata.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that:
    /// - `data` points to initialized storage matching `shape`, `strides`, and
    ///   `dtype`, residing on the device described by `device`;
    /// - that storage remains valid for the lifetime `'a`;
    /// - the C API consumes the resulting [`DLManagedTensor`] (including its
    ///   `shape`/`strides` pointers) only for the duration of the FFI call and
    ///   does not retain it afterward — cuVS upholds this.
    pub unsafe fn from_raw_parts(
        data: *mut std::ffi::c_void,
        device: ffi::DLDevice,
        shape: &[i64],
        strides: Option<&[i64]>,
        dtype: ffi::DLDataType,
    ) -> std::result::Result<Self, DLPackError> {
        if let Some(s) = strides
            && s.len() != shape.len()
        {
            return Err(DLPackError::StridesLenMismatch { ndim: shape.len(), strides: s.len() });
        }
        Ok(Self {
            data,
            device,
            dtype,
            shape: shape.iter().copied().collect(),
            strides: strides.map(|s| s.iter().copied().collect()),
            _marker: PhantomData,
        })
    }

    /// Build a [`DLManagedTensor`] for an FFI call.
    ///
    /// DLPack stores `shape` and `strides` as mutable pointers, so this method
    /// casts pointers derived from `&self` to `*mut` for C ABI compatibility.
    /// The callee must treat them as `const`.
    pub(crate) fn to_c(&self) -> ManagedTensorRef<'_> {
        ManagedTensorRef {
            inner: ffi::DLManagedTensor {
                dl_tensor: ffi::DLTensor {
                    data: self.data,
                    device: self.device,
                    ndim: self.shape.len() as i32,
                    dtype: self.dtype,
                    shape: self.shape.as_ptr() as *mut _,
                    strides: self
                        .strides
                        .as_ref()
                        .map_or(std::ptr::null_mut(), |s| s.as_ptr() as *mut _),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            },
            _borrow: PhantomData,
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape of the tensor.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Strides, if non-contiguous. `None` means row-major contiguous.
    pub fn strides(&self) -> Option<&[i64]> {
        self.strides.as_deref()
    }

    /// Element data type.
    pub fn dtype(&self) -> ffi::DLDataType {
        self.dtype
    }

    /// Device where the data resides.
    pub fn device(&self) -> ffi::DLDevice {
        self.device
    }
}

impl fmt::Debug for DLTensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DLTensorView")
            .field("shape", &self.shape.as_slice())
            .field("strides", &self.strides.as_deref())
            .finish()
    }
}

/// A non-owning, writable DLPack tensor view.
#[must_use]
pub struct DLTensorViewMut<'a> {
    base: DLTensorView<'a>,
    _unique: PhantomData<&'a mut ()>,
}

impl<'a> DLTensorViewMut<'a> {
    /// Construct a writable DLPack view from raw tensor metadata.
    ///
    /// # Safety
    ///
    /// In addition to the [`DLTensorView::from_raw_parts`] invariants, the
    /// caller must guarantee the storage is exclusively writable for `'a`.
    pub unsafe fn from_raw_parts(
        data: *mut std::ffi::c_void,
        device: ffi::DLDevice,
        shape: &[i64],
        strides: Option<&[i64]>,
        dtype: ffi::DLDataType,
    ) -> std::result::Result<Self, DLPackError> {
        Ok(Self {
            base: unsafe { DLTensorView::from_raw_parts(data, device, shape, strides, dtype)? },
            _unique: PhantomData,
        })
    }
}

impl<'a> Deref for DLTensorViewMut<'a> {
    type Target = DLTensorView<'a>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl fmt::Debug for DLTensorViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DLTensorViewMut")
            .field("shape", &self.base.shape.as_slice())
            .field("strides", &self.base.strides.as_deref())
            .finish()
    }
}

impl AsDlTensor for DLTensorView<'_> {
    fn as_dl_tensor(&self) -> std::result::Result<DLTensorView<'_>, DLPackError> {
        Ok(DLTensorView {
            data: self.data,
            device: self.device,
            dtype: self.dtype,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _marker: PhantomData,
        })
    }
}

impl AsDlTensor for DLTensorViewMut<'_> {
    fn as_dl_tensor(&self) -> std::result::Result<DLTensorView<'_>, DLPackError> {
        // Reborrow the read-only base as a fresh view bound to `&self`.
        self.base.as_dl_tensor()
    }
}

impl AsDlTensorMut for DLTensorViewMut<'_> {
    fn as_dl_tensor_mut(&mut self) -> std::result::Result<DLTensorViewMut<'_>, DLPackError> {
        Ok(DLTensorViewMut { base: self.base.as_dl_tensor()?, _unique: PhantomData })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> DLDevice {
        DLDevice { device_type: DLDeviceType::kDLCPU, device_id: 0 }
    }

    #[test]
    fn to_c_translates_contiguous_metadata() {
        let data = [0.0f32; 6];
        let view = unsafe {
            DLTensorView::from_raw_parts(
                data.as_ptr() as *mut _,
                cpu(),
                &[2, 3],
                None,
                f32::dl_dtype(),
            )
        }
        .unwrap();

        let managed = view.to_c();
        let t = &managed.inner.dl_tensor;

        assert_eq!(t.ndim, 2);
        assert_eq!(t.data as *const f32, data.as_ptr());
        assert_eq!(t.dtype.code, DLDataTypeCode::kDLFloat as u8);
        assert_eq!(t.dtype.bits, 32);
        assert_eq!(t.dtype.lanes, 1);
        assert_eq!(t.byte_offset, 0);
        assert!(managed.inner.manager_ctx.is_null());
        assert!(managed.inner.deleter.is_none());

        assert_eq!(unsafe { std::slice::from_raw_parts(t.shape, 2) }, &[2, 3]);
        assert!(t.strides.is_null());
    }

    #[test]
    fn to_c_preserves_explicit_strides() {
        let data = [0.0f32; 6];
        let view = unsafe {
            DLTensorView::from_raw_parts(
                data.as_ptr() as *mut _,
                cpu(),
                &[2, 3],
                Some(&[1, 2]),
                f32::dl_dtype(),
            )
        }
        .unwrap();

        let managed = view.to_c();
        let t = &managed.inner.dl_tensor;

        assert_eq!(unsafe { std::slice::from_raw_parts(t.shape, 2) }, &[2, 3]);
        assert!(!t.strides.is_null());
        assert_eq!(unsafe { std::slice::from_raw_parts(t.strides, 2) }, &[1, 2]);
    }

    #[test]
    fn from_raw_parts_rejects_mismatched_strides_len() {
        let data = [0.0f32; 6];
        let err = unsafe {
            DLTensorView::from_raw_parts(
                data.as_ptr() as *mut _,
                cpu(),
                &[2, 3],
                Some(&[1]),
                f32::dl_dtype(),
            )
        }
        .unwrap_err();

        assert!(matches!(err, DLPackError::StridesLenMismatch { ndim: 2, strides: 1 }));
    }
}
