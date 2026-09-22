/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::{CString, NulError};
use std::io::{Write, stderr};
use std::path::Path;

use crate::error::{LibraryError, check_cuvs};

/// Run an FFI constructor that reports its result through an out-pointer.
///
/// # Safety
///
/// `f` must initialize the out-pointer whenever it reports success.
pub(crate) unsafe fn init_handle<H>(
    f: impl FnOnce(*mut H) -> ffi::cuvsError_t,
) -> Result<H, LibraryError> {
    let mut out = std::mem::MaybeUninit::<H>::uninit();
    check_cuvs(f(out.as_mut_ptr()))?;
    Ok(unsafe { out.assume_init() })
}

pub(crate) fn report_drop_failure(what: &str, err: &LibraryError) {
    let _ = writeln!(stderr(), "cuvs: failed to destroy {what}: {err:?}");
}

pub(crate) fn path_to_cstring(path: &Path) -> Result<CString, NulError> {
    CString::new(path.as_os_str().as_encoded_bytes())
}
