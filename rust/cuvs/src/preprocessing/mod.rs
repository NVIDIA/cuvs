/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Preprocessing utilities for cuVS datasets.
//!
//! Currently this exposes the [`quantize`] module, which provides quantizers
//! that compress floating-point datasets into more compact representations.

pub mod quantize;
