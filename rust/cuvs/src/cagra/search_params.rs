/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::{Result, check_cuvs};
use std::fmt;
use std::io::{Write, stderr};

pub type SearchAlgo = ffi::cuvsCagraSearchAlgo;
pub type HashMode = ffi::cuvsCagraHashMode;

/// Supplemental parameters to search CAGRA index
pub struct SearchParams(pub ffi::cuvsCagraSearchParams_t);

impl SearchParams {
    /// Returns a new SearchParams object
    pub fn new() -> Result<SearchParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsCagraSearchParams_t>::uninit();
            check_cuvs(ffi::cuvsCagraSearchParamsCreate(params.as_mut_ptr()))?;
            Ok(SearchParams(params.assume_init()))
        }
    }

    /// Maximum number of queries to search at the same time (batch size). Auto select when 0
    pub fn set_max_queries(self, max_queries: usize) -> SearchParams {
        unsafe {
            (*self.0).max_queries = max_queries;
        }
        self
    }

    /// Number of intermediate search results retained during the search.
    /// This is the main knob to adjust trade off between accuracy and search speed.
    /// Higher values improve the search accuracy
    pub fn set_itopk_size(self, itopk_size: usize) -> SearchParams {
        unsafe {
            (*self.0).itopk_size = itopk_size;
        }
        self
    }

    /// Upper limit of search iterations. Auto select when 0.
    pub fn set_max_iterations(self, max_iterations: usize) -> SearchParams {
        unsafe {
            (*self.0).max_iterations = max_iterations;
        }
        self
    }

    /// Which search implementation to use.
    pub fn set_algo(self, algo: SearchAlgo) -> SearchParams {
        unsafe {
            (*self.0).algo = algo;
        }
        self
    }

    /// Number of threads used to calculate a single distance. 4, 8, 16, or 32.
    pub fn set_team_size(self, team_size: usize) -> SearchParams {
        unsafe {
            (*self.0).team_size = team_size;
        }
        self
    }

    /// Lower limit of search iterations.
    pub fn set_min_iterations(self, min_iterations: usize) -> SearchParams {
        unsafe {
            (*self.0).min_iterations = min_iterations;
        }
        self
    }

    /// Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.
    pub fn set_thread_block_size(self, thread_block_size: usize) -> SearchParams {
        unsafe {
            (*self.0).thread_block_size = thread_block_size;
        }
        self
    }

    /// Hashmap type. Auto selection when AUTO.
    pub fn set_hashmap_mode(self, hashmap_mode: HashMode) -> SearchParams {
        unsafe {
            (*self.0).hashmap_mode = hashmap_mode;
        }
        self
    }

    /// Lower limit of hashmap bit length. More than 8.
    pub fn set_hashmap_min_bitlen(self, hashmap_min_bitlen: usize) -> SearchParams {
        unsafe {
            (*self.0).hashmap_min_bitlen = hashmap_min_bitlen;
        }
        self
    }

    /// Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    pub fn set_hashmap_max_fill_rate(self, hashmap_max_fill_rate: f32) -> SearchParams {
        unsafe {
            (*self.0).hashmap_max_fill_rate = hashmap_max_fill_rate;
        }
        self
    }

    /// Number of iterations of initial random seed node selection. 1 or more.
    pub fn set_num_random_samplings(self, num_random_samplings: u32) -> SearchParams {
        unsafe {
            (*self.0).num_random_samplings = num_random_samplings;
        }
        self
    }

    /// Bit mask used for initial random seed node selection.
    pub fn set_rand_xor_mask(self, rand_xor_mask: u64) -> SearchParams {
        unsafe {
            (*self.0).rand_xor_mask = rand_xor_mask;
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraSearchParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsCagraSearchParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_params() {
        let params = SearchParams::new().unwrap().set_itopk_size(128);

        unsafe {
            assert_eq!((*params.0).itopk_size, 128);
        }
    }
}
