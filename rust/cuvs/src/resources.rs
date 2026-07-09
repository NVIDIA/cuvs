/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::{Error, Result, check_cuvs};
use std::ffi::CString;
use std::io::{Write, stderr};
use std::path::Path;
use std::time::Duration;

/// Resources are objects that are shared between function calls,
/// and includes things like CUDA streams, cuBLAS handles and other
/// resources that are expensive to create.
#[derive(Debug)]
pub struct Resources(pub ffi::cuvsResources_t);

impl Resources {
    /// Returns a new Resources object
    pub fn new() -> Result<Resources> {
        let mut res: ffi::cuvsResources_t = 0;
        unsafe {
            check_cuvs(ffi::cuvsResourcesCreate(&mut res))?;
        }
        Ok(Resources(res))
    }

    /// Returns a new `Resources` object whose memory allocations are tracked
    /// and written as CSV samples to `csv_path` from a background thread.
    ///
    /// The handle wraps all reachable memory resources (host, pinned, managed,
    /// device, workspace, large_workspace) with allocation-tracking adaptors
    /// and replaces the global host and device memory resources for the
    /// lifetime of the handle. The CSV reporter is stopped and the global
    /// memory resources are restored when the handle is dropped.
    ///
    /// `sample_interval` controls the minimum time between successive CSV
    /// samples; when `None`, the C++ default of 10 ms is used.
    pub fn with_memory_tracking(
        csv_path: impl AsRef<Path>,
        sample_interval: Option<Duration>,
    ) -> Result<Resources> {
        let c_path =
            CString::new(csv_path.as_ref().as_os_str().as_encoded_bytes()).map_err(|e| {
                Error::InvalidArgument(format!("csv_path contains an interior NUL byte: {}", e))
            })?;
        let sample_interval_ms =
            sample_interval.unwrap_or(Duration::from_millis(10)).as_millis() as i64;
        let mut res: ffi::cuvsResources_t = 0;
        unsafe {
            check_cuvs(ffi::cuvsResourcesCreateWithMemoryTracking(
                &mut res,
                c_path.as_ptr(),
                sample_interval_ms,
            ))?;
        }
        Ok(Resources(res))
    }

    /// Sets the current cuda stream
    ///
    /// # Safety
    ///
    /// `stream` must be a valid CUDA stream that remains valid for as long as it
    /// is used by this resources handle.
    pub unsafe fn set_cuda_stream(&self, stream: ffi::cudaStream_t) -> Result<()> {
        unsafe { check_cuvs(ffi::cuvsStreamSet(self.0, stream)) }
    }

    /// Gets the current cuda stream
    pub fn get_cuda_stream(&self) -> Result<ffi::cudaStream_t> {
        unsafe {
            let mut stream = std::mem::MaybeUninit::<ffi::cudaStream_t>::uninit();
            check_cuvs(ffi::cuvsStreamGet(self.0, stream.as_mut_ptr()))?;
            Ok(stream.assume_init())
        }
    }

    /// Syncs the current cuda stream
    pub fn sync_stream(&self) -> Result<()> {
        unsafe { check_cuvs(ffi::cuvsStreamSync(self.0)) }
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = check_cuvs(ffi::cuvsResourcesDestroy(self.0)) {
                write!(stderr(), "failed to call cuvsResourcesDestroy {:?}", e)
                    .expect("failed to write to stderr");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resources_create() {
        let _ = Resources::new();
    }

    #[test]
    fn test_resources_with_memory_tracking() {
        let dir = tempfile::tempdir().unwrap();
        let csv = dir.path().join("alloc.csv");
        {
            let _r = Resources::with_memory_tracking(&csv, Some(Duration::from_millis(2)))
                .expect("with_memory_tracking should succeed");
            // closing _r at end of scope flushes the CSV reporter and
            // restores the global host/device memory resources.
        }
        let meta = std::fs::metadata(&csv).expect("csv file should exist after drop");
        // at minimum, the header row should have been written before drop
        assert!(meta.len() > 0, "tracking csv should be non-empty (got {} bytes)", meta.len());
    }
}
