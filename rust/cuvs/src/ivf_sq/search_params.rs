/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::{Result, check_cuvs};
use std::fmt;
use std::io::{Write, stderr};

/// Supplemental parameters to search IVF-SQ index
pub struct SearchParams(pub ffi::cuvsIvfSqSearchParams_t);

impl SearchParams {
    /// Returns a new SearchParams object
    pub fn new() -> Result<SearchParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsIvfSqSearchParams_t>::uninit();
            check_cuvs(ffi::cuvsIvfSqSearchParamsCreate(params.as_mut_ptr()))?;
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfSqSearchParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsIvfSqSearchParamsDestroy {:?}", e)
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
