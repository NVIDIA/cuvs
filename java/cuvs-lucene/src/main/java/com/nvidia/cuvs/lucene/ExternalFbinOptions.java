/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.util.Objects;

/** Validation and scheduling policy for an immutable external-FBIN bulk build. */
public record ExternalFbinOptions(ExternalFbinBuildValidation validation, long scanHeadStartBytes) {

  public ExternalFbinOptions {
    Objects.requireNonNull(validation, "validation");
    if (scanHeadStartBytes < 0L) {
      throw new IllegalArgumentException("scanHeadStartBytes must be non-negative");
    }
    if (validation == ExternalFbinBuildValidation.TRUSTED_IMMUTABLE && scanHeadStartBytes != 0L) {
      throw new IllegalArgumentException(
          "scanHeadStartBytes requires PREFETCH or VERIFY_SHA256 validation");
    }
  }

  /** Uses complete-file SHA-256 verification with no artificial graph-start delay. */
  public static ExternalFbinOptions verifySha256() {
    return new ExternalFbinOptions(ExternalFbinBuildValidation.VERIFY_SHA256, 0L);
  }
}
