/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuVS library version query.

use crate::error::{LibraryError, check_cuvs};
use crate::ffi;

/// Returns the cuVS library version as `(major, minor, patch)`.
pub fn version() -> Result<(u16, u16, u16), LibraryError> {
    let mut major: u16 = 0;
    let mut minor: u16 = 0;
    let mut patch: u16 = 0;

    // SAFETY:
    // - All three pointers are valid, aligned `u16` locals.
    let status = unsafe { ffi::cuvsVersionGet(&mut major, &mut minor, &mut patch) };
    check_cuvs(status)?;
    Ok((major, minor, patch))
}
