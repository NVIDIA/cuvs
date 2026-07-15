/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CAGRA: a graph-based approximate nearest neighbors algorithm with
//! state-of-the-art query throughput for both small and large batch sizes.
//!
//! Build an [`Index`] from a dataset, then [`search`](Index::search) it with
//! device-resident queries and output buffers. Tensors are passed through the
//! [`AsDlTensor`](crate::AsDlTensor) /
//! [`AsDlTensorMut`](crate::AsDlTensorMut) traits; see the
//! [`dlpack`](crate::dlpack) module for the tensor model and `examples/cagra.rs`
//! for a complete, runnable example.

mod index;
mod index_params;
mod search_params;

pub use index::Index;
pub use index_params::{BuildAlgo, CompressionParams, IndexParams};
pub use search_params::{HashMode, SearchAlgo, SearchParams};
