/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! IVF-Flat: an inverted-file index over uncompressed ("flat") vectors. It
//! partitions the dataset into `n_lists` clusters and, at query time, scans only
//! the `n_probes` closest clusters — a simple knob to trade recall for speed.
//!
//! Build an [`Index`] from a dataset, then [`search`](Index::search) it with
//! device-resident queries and output buffers. Tensors are passed through the
//! [`AsDlTensor`](crate::AsDlTensor) /
//! [`AsDlTensorMut`](crate::AsDlTensorMut) traits; see the
//! [`dlpack`](crate::dlpack) module for the tensor model and `examples/cagra.rs`
//! for the same build/search workflow.

mod index;
mod index_params;
mod search_params;

pub use index::Index;
pub use index_params::IndexParams;
pub use search_params::SearchParams;
