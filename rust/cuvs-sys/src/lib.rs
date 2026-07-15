/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw FFI bindings to libcuvs_c.

use std::os::raw::c_uint;

/// Opaque CUDA stream handle used by the current cuVS C ABI.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut CUstream_st;

/// Temporary ABI shim for `cudaDataType_t` while the cuVS C API exposes CUDA types.
/// TODO: Remove this once the cuVS C API removes `cudaDataType_t` reliance.
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct cudaDataType_t(c_uint);

impl cudaDataType_t {
    pub const CUDA_R_32F: Self = Self(0);
    pub const CUDA_R_16F: Self = Self(2);
    pub const CUDA_R_8I: Self = Self(3);
    pub const CUDA_R_8U: Self = Self(8);

    pub const fn from_raw(value: c_uint) -> Self {
        Self(value)
    }

    pub const fn as_raw(self) -> c_uint {
        self.0
    }
}

// Bindings are pre-generated and checked in at src/bindings.rs.
// Use `rust/scripts/generate-bindings.sh` to regenerate them.
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, unused_attributes)]
mod bindings;

// Bindgen cannot derive these for cuvsIvfPqSearchParams once cudaDataType_t is
// supplied by this crate instead of generated from CUDA headers.
impl Copy for bindings::cuvsIvfPqSearchParams {}

impl Clone for bindings::cuvsIvfPqSearchParams {
    fn clone(&self) -> Self {
        *self
    }
}

impl std::fmt::Debug for bindings::cuvsIvfPqSearchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("cuvsIvfPqSearchParams")
            .field("n_probes", &self.n_probes)
            .field("lut_dtype", &self.lut_dtype)
            .field("internal_distance_dtype", &self.internal_distance_dtype)
            .field("coarse_search_dtype", &self.coarse_search_dtype)
            .field("max_internal_batch_size", &self.max_internal_batch_size)
            .field("preferred_shmem_carveout", &self.preferred_shmem_carveout)
            .finish()
    }
}

pub use bindings::*;
