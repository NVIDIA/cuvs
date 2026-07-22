/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The tiered index couples a brute-force tier that absorbs incremental inserts
//! with an ANN tier (CAGRA by default; IVF-Flat and IVF-PQ are also available).
//!
//! Vectors added with [`Index::extend`] land in the brute-force tier and are
//! immediately searchable — even before the ANN tier has been (re)built. Once
//! the incremental tier exceeds `min_ann_rows`, the ANN tier is built (or
//! rebuilt on extend when `create_ann_index_on_extend` is set).
//!
//! Build an [`Index`] from a dataset, [`extend`](Index::extend) it with new
//! vectors, then [`search`](Index::search) it with device-resident queries and
//! output buffers — the [`SearchParams`] variant must match the ANN backend the
//! index was built with. Tensors are passed through the
//! [`AsDlTensor`](crate::AsDlTensor) /
//! [`AsDlTensorMut`](crate::AsDlTensorMut) traits; see the
//! [`dlpack`](crate::dlpack) module for the tensor model and `examples/cagra.rs`
//! for the same build/search workflow.
//!
//! The C API does not provide serialize/deserialize for the tiered index, so
//! this module does not expose persistence.

mod index;
mod index_params;
mod search_params;

pub use index::Index;
pub use index_params::{AnnAlgo, IndexParams};
pub use search_params::SearchParams;
