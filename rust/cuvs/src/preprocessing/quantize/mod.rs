/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Dataset quantizers.
//!
//! Quantizers compress a floating-point dataset into a more compact
//! representation. The [`scalar`] quantizer maps an interval of the input
//! float range onto the full range of an 8-bit integer.
//!
//! Train a [`scalar::Quantizer`] on a dataset, then
//! [`transform`](scalar::Quantizer::transform) a float matrix into int8 and
//! [`inverse_transform`](scalar::Quantizer::inverse_transform) it back to an
//! approximation of the original. Tensors are passed through the
//! [`AsDlTensor`](crate::AsDlTensor) /
//! [`AsDlTensorMut`](crate::AsDlTensorMut) traits; see the
//! [`dlpack`](crate::dlpack) module for the tensor model and `examples/cagra.rs`
//! for a device-tensor adapter.
//!
//! The binary and product (PQ) quantizers exposed by the cuVS C API are not
//! yet wrapped in Rust; they are intended to be added in follow-up
//! contributions.

pub mod scalar;
