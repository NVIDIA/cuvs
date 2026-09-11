/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

/** Controls validation and sequential read-ahead for an immutable external FBIN build. */
public enum ExternalFbinBuildValidation {
  /**
   * Trusts that the registered file was previously verified and is protected by immutable,
   * content-addressed storage. Performs structural checks but no full payload scan.
   */
  TRUSTED_IMMUTABLE,

  /**
   * Sequentially scans the referenced payload alongside graph construction to provide controlled
   * read-ahead, but trusts the precomputed digest.
   */
  PREFETCH,

  /**
   * Sequentially hashes the complete FBIN alongside graph construction and fails the build if its
   * SHA-256 differs. This also acts as read-ahead, but may extend the critical path on slow storage.
   */
  VERIFY_SHA256
}
