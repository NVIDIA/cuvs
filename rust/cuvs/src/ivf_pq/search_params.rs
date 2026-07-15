/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::{Result, check_cuvs};
use std::fmt;
use std::io::{Write, stderr};

pub use ffi::cudaDataType_t;

/// Supplemental parameters to search IvfPq index
pub struct SearchParams(pub ffi::cuvsIvfPqSearchParams_t);

impl SearchParams {
    /// Returns a new SearchParams object
    pub fn new() -> Result<SearchParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsIvfPqSearchParams_t>::uninit();
            check_cuvs(ffi::cuvsIvfPqSearchParamsCreate(params.as_mut_ptr()))?;
            Ok(SearchParams(params.assume_init()))
        }
    }

    /// The number of clusters to search.
    pub fn set_n_probes(self, n_probes: u32) -> SearchParams {
        unsafe {
            (*self.0).n_probes = n_probes;
        }
        self
    }

    /// Data type of look up table to be created dynamically at search
    /// time. The use of low-precision types reduces the amount of shared
    /// memory required at search time, so fast shared memory kernels can
    /// be used even for datasets with large dimansionality. Note that
    /// the recall is slightly degraded when low-precision type is
    /// selected.
    pub fn set_lut_dtype(self, lut_dtype: cudaDataType_t) -> SearchParams {
        unsafe {
            (*self.0).lut_dtype = lut_dtype;
        }
        self
    }

    /// Storage data type for distance/similarity computation.
    pub fn set_internal_distance_dtype(
        self,
        internal_distance_dtype: cudaDataType_t,
    ) -> SearchParams {
        unsafe {
            (*self.0).internal_distance_dtype = internal_distance_dtype;
        }
        self
    }
}

impl fmt::Debug for SearchParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug trait here, default value will show the pointer address
        // for the inner params object which isn't that useful.
        write!(f, "SearchParams {{ params: {:?} }}", unsafe { *self.0 })
    }
}

impl Drop for SearchParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfPqSearchParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsIvfPqSearchParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_params() {
        let params = SearchParams::new().unwrap().set_n_probes(128);

        unsafe {
            assert_eq!((*params.0).n_probes, 128);
        }
    }
}
