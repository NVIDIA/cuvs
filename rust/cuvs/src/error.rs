/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fmt;

#[derive(Debug, Clone)]
pub struct CuvsError {
    code: ffi::cuvsError_t,
    text: String,
}

#[derive(Debug, Clone)]
pub enum Error {
    CuvsError(CuvsError),
    /// Tensor conversion into DLPack metadata failed.
    DLPack(crate::dlpack::DLPackError),
    /// The caller passed an argument that could not be forwarded to the C API
    /// (e.g. a filename containing an interior NUL byte or invalid UTF-8).
    InvalidArgument(String),
}

impl std::error::Error for Error {}
impl std::error::Error for CuvsError {}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::CuvsError(cuvs_error) => write!(f, "cuvsError={:?}", cuvs_error),
            Error::DLPack(err) => write!(f, "DLPack error: {}", err),
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {}", msg),
        }
    }
}

impl fmt::Display for CuvsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}:{}", self.code, self.text)
    }
}

/// Simple wrapper to convert a cuvsError_t into a Result
pub fn check_cuvs(err: ffi::cuvsError_t) -> Result<()> {
    match err {
        ffi::cuvsError_t::CUVS_SUCCESS => Ok(()),
        _ => {
            // get a description of the error from cuvs
            let cstr = unsafe {
                let text_ptr = ffi::cuvsGetLastErrorText();
                std::ffi::CStr::from_ptr(text_ptr)
            };
            let text = std::string::String::from_utf8_lossy(cstr.to_bytes()).to_string();

            Err(Error::CuvsError(CuvsError { code: err, text }))
        }
    }
}

impl From<crate::dlpack::DLPackError> for Error {
    fn from(err: crate::dlpack::DLPackError) -> Self {
        Self::DLPack(err)
    }
}
