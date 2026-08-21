/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU resource management with RAII semantics.

use crate::error::{LibraryError, check_cuvs};
use std::ffi::CString;
use std::io::{Write, stderr};
use std::path::Path;
use std::time::Duration;

type Result<T> = std::result::Result<T, ResourcesError>;

/// Error type for resource operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ResourcesError {
    /// The cuVS C library reported a failure.
    #[error(transparent)]
    Library(#[from] LibraryError),
    /// A CSV path contained an interior NUL byte.
    #[error("CSV path contains an interior NUL byte")]
    InvalidPath(#[from] std::ffi::NulError),
}

/// Resources are objects that are shared between function calls,
/// and includes things like CUDA streams, cuBLAS handles and other
/// resources that are expensive to create.
#[derive(Debug)]
pub struct Resources {
    handle: ffi::cuvsResources_t,
}

impl Resources {
    /// Creates a new resources handle bound to the current CUDA device.
    pub fn new() -> Result<Resources> {
        let mut handle: ffi::cuvsResources_t = 0;
        check_cuvs(unsafe { ffi::cuvsResourcesCreate(&mut handle) })?;
        Ok(Resources { handle })
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
        let c_path = CString::new(csv_path.as_ref().as_os_str().as_encoded_bytes())?;
        let sample_interval_ms =
            sample_interval.unwrap_or(Duration::from_millis(10)).as_millis() as i64;
        let mut handle: ffi::cuvsResources_t = 0;
        check_cuvs(unsafe {
            ffi::cuvsResourcesCreateWithMemoryTracking(
                &mut handle,
                c_path.as_ptr(),
                sample_interval_ms,
            )
        })?;
        Ok(Resources { handle })
    }

    /// Creates a resources handle that enqueues work on `stream` instead of the
    /// default internal stream.
    ///
    /// The stream is bound once, at construction.
    ///
    /// # Safety
    ///
    /// `stream` must be a valid CUDA stream for the current device and must
    /// remain valid for as long as this handle uses it.
    pub unsafe fn with_stream(stream: ffi::cudaStream_t) -> Result<Resources> {
        let res = Resources::new()?;
        // SAFETY: the caller guarantees `stream` is valid for this device and
        // outlives the handle.
        check_cuvs(unsafe { ffi::cuvsStreamSet(res.handle, stream) })?;
        Ok(res)
    }

    /// Returns the current CUDA stream associated with this handle.
    pub fn stream(&self) -> Result<ffi::cudaStream_t> {
        unsafe {
            let mut stream = std::mem::MaybeUninit::<ffi::cudaStream_t>::uninit();
            check_cuvs(ffi::cuvsStreamGet(self.handle, stream.as_mut_ptr()))?;
            Ok(stream.assume_init())
        }
    }

    /// Blocks until all operations on the current CUDA stream have completed.
    pub fn sync_stream(&self) -> Result<()> {
        check_cuvs(unsafe { ffi::cuvsStreamSync(self.handle) })?;
        Ok(())
    }

    /// Raw handle for FFI calls in other modules.
    pub(crate) fn handle(&self) -> ffi::cuvsResources_t {
        self.handle
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsResourcesDestroy(self.handle) }) {
            write!(stderr(), "failed to call cuvsResourcesDestroy {:?}", e)
                .expect("failed to write to stderr");
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
}
