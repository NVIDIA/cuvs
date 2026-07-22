/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuVS: Rust bindings for Vector Search on the GPU
//!
//! This crate provides Rust bindings for cuVS, allowing you to run
//! approximate nearest neighbors search on the GPU.
extern crate cuvs_sys as ffi;

pub mod brute_force;
pub mod cagra;
pub mod cluster;
pub mod distance;
pub mod distance_type;
pub mod dlpack;
mod error;
pub mod ivf_flat;
pub mod ivf_pq;
pub mod refine;
mod resources;
#[cfg(test)]
pub(crate) mod test_utils;
pub mod vamana;

pub use dlpack::{AsDlTensor, AsDlTensorMut, DLPackError, DLTensorView, DLTensorViewMut, DType};
pub use error::{Error, Result};
pub use resources::Resources;
