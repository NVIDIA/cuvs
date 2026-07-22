/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared filter payloads for nearest-neighbor search APIs.

use std::marker::PhantomData;

use crate::dlpack::{AsDlTensor, DLPackError, DLTensorView};

/// Error returned when constructing an invalid filter payload.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FilterError {
    /// The filter tensor must be a 1-D vector of packed 32-bit words.
    #[error("filter must be a 1-D tensor")]
    InvalidRank,
    /// The filter tensor must be in device-accessible memory.
    #[error("filter must use device-accessible memory")]
    InvalidDevice,
    /// The filter tensor must use scalar `u32` words.
    #[error("filter must use scalar 32-bit unsigned words (`u32`)")]
    InvalidDType,
    /// The filter tensor must be contiguous.
    #[error("filter must be contiguous")]
    NonContiguous,
    /// The source tensor could not be converted to a DLPack view.
    #[error(transparent)]
    Conversion(#[from] DLPackError),
}

/// Marker for a row-level bitset filter.
pub enum Bitset {}

/// Marker for a per-query bitmap filter.
pub enum Bitmap {}

mod sealed {
    pub trait Sealed {}
}

/// Kind of packed filter payload.
///
/// This trait is sealed; only [`Bitset`] and [`Bitmap`] implement it.
pub trait FilterKind: sealed::Sealed {
    #[doc(hidden)]
    const FILTER_TYPE: ffi::cuvsFilterType;
}

impl sealed::Sealed for Bitset {}

impl FilterKind for Bitset {
    const FILTER_TYPE: ffi::cuvsFilterType = ffi::cuvsFilterType::BITSET;
}

impl sealed::Sealed for Bitmap {}

impl FilterKind for Bitmap {
    const FILTER_TYPE: ffi::cuvsFilterType = ffi::cuvsFilterType::BITMAP;
}

/// Packed filter words used to include or exclude rows during search.
pub struct Filter<'a, K: FilterKind> {
    tensor: DLTensorView<'a>,
    _kind: PhantomData<K>,
}

impl<'a, K: FilterKind> Filter<'a, K> {
    /// Creates a packed filter borrowing `filter_words`.
    pub fn new<T>(filter_words: &'a T) -> Result<Self, FilterError>
    where
        T: AsDlTensor + ?Sized,
    {
        let tensor = filter_words.as_dl_tensor()?;
        if tensor.ndim() != 1 {
            return Err(FilterError::InvalidRank);
        }
        if !matches!(
            tensor.device().device_type,
            ffi::DLDeviceType::kDLCUDA
                | ffi::DLDeviceType::kDLCUDAHost
                | ffi::DLDeviceType::kDLCUDAManaged
        ) {
            return Err(FilterError::InvalidDevice);
        }
        if !is_contiguous_1d(tensor.strides()) {
            return Err(FilterError::NonContiguous);
        }

        let dtype = tensor.dtype();
        if dtype.code != ffi::DLDataTypeCode::kDLUInt as u8 || dtype.bits != 32 || dtype.lanes != 1
        {
            return Err(FilterError::InvalidDType);
        }

        Ok(Self { tensor, _kind: PhantomData })
    }
}

fn is_contiguous_1d(strides: Option<&[i64]>) -> bool {
    matches!(strides, None | Some([1]))
}

pub(crate) fn with_filter<K: FilterKind, R>(
    filter: Option<&Filter<'_, K>>,
    call: impl FnOnce(ffi::cuvsFilter) -> R,
) -> R {
    match filter {
        Some(filter) => {
            let mut managed = filter.tensor.to_c();
            call(ffi::cuvsFilter { addr: managed.as_mut_ptr() as usize, type_: K::FILTER_TYPE })
        }
        None => call(ffi::cuvsFilter { addr: 0, type_: ffi::cuvsFilterType::NO_FILTER }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_view<'a>(words: &'a mut [u32], shape: &[i64]) -> DLTensorView<'a> {
        unsafe {
            DLTensorView::from_raw_parts(
                words.as_mut_ptr().cast(),
                ffi::DLDevice { device_type: ffi::DLDeviceType::kDLCPU, device_id: 0 },
                shape,
                None,
                ffi::DLDataType { code: ffi::DLDataTypeCode::kDLUInt as u8, bits: 32, lanes: 1 },
            )
            .unwrap()
        }
    }

    #[test]
    fn rejects_non_vector_filters() {
        let mut words = [0u32];
        let view = host_view(&mut words, &[1, 1]);
        assert!(matches!(Filter::<Bitset>::new(&view), Err(FilterError::InvalidRank)));
    }

    #[test]
    fn rejects_host_filters() {
        let mut words = [0u32];
        let view = host_view(&mut words, &[1]);
        assert!(matches!(Filter::<Bitset>::new(&view), Err(FilterError::InvalidDevice)));
    }

    #[test]
    fn explicit_unit_stride_is_contiguous() {
        assert!(is_contiguous_1d(None));
        assert!(is_contiguous_1d(Some(&[1])));
        assert!(!is_contiguous_1d(Some(&[2])));
    }

    #[test]
    fn no_filter_uses_the_c_api_sentinel() {
        with_filter::<Bitset, _>(None, |filter| {
            assert_eq!(filter.addr, 0);
            assert_eq!(filter.type_, ffi::cuvsFilterType::NO_FILTER);
        });
    }
}
