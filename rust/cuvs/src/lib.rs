/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuVS: Rust bindings for Vector Search on the GPU
//!
//! This crate provides Rust bindings for cuVS, allowing you to run
//! approximate nearest neighbors search on the GPU.
extern crate cuvs_sys as ffi;

pub mod cluster;
pub mod distance;
pub mod dlpack;
mod error;
pub mod neighbors;
mod resources;
#[cfg(test)]
pub(crate) mod test_utils;

pub use dlpack::{AsDlTensor, AsDlTensorMut, DLPackError, DLTensorView, DLTensorViewMut, DType};
pub use error::LibraryError;
pub use resources::Resources;

// Compile the Rust code blocks in the top-level README as doctests so the
// documented examples can't drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct ReadmeDocTests;
