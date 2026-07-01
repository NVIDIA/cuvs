/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Low-level error handling shared by every cuVS module.
//!
//! [`check_cuvs`] turns a raw `cuvsError_t` status into a [`LibraryError`], which
//! each module's error type wraps via `#[from]`.

use std::borrow::Cow;

/// A failure reported by the cuVS C library.
///
/// Carries the message captured from `cuvsGetLastErrorText` at the point of
/// failure. Every module's error type wraps this via `#[from]`.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct LibraryError(Cow<'static, str>);

/// Converts a `cuvsError_t` status into a [`LibraryError`].
///
/// On failure the thread-local error text is captured immediately, before any
/// subsequent FFI call can overwrite it.
pub(crate) fn check_cuvs(status: ffi::cuvsError_t) -> Result<(), LibraryError> {
    match status {
        ffi::cuvsError_t::CUVS_SUCCESS => Ok(()),
        _ => {
            // SAFETY: `cuvsGetLastErrorText` returns either NULL or a pointer to
            // thread-local storage valid until the next FFI call; copy it now.
            let text: Cow<'static, str> = unsafe {
                let text_ptr = ffi::cuvsGetLastErrorText();
                if text_ptr.is_null() {
                    Cow::Borrowed("unknown cuVS error")
                } else {
                    let cstr = std::ffi::CStr::from_ptr(text_ptr);
                    Cow::Owned(String::from_utf8_lossy(cstr.to_bytes()).into_owned())
                }
            };
            Err(LibraryError(text))
        }
    }
}
