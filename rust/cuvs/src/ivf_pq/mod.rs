/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! IVF-PQ: an inverted-file index that product-quantizes the vectors. Like
//! IVF-Flat it partitions the dataset into `n_lists` clusters and scans the
//! `n_probes` closest at query time, but compresses each vector into `pq_dim`
//! codes of `pq_bits` bits — much smaller, slightly less accurate.
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
