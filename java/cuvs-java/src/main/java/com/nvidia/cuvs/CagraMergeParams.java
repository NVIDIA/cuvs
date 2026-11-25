/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

public class CagraMergeParams {

  private final CagraIndexParams outputIndexParams;

  /**
   * Constructs a CagraMergeParams with the given output index parameters.
   *
   * @param outputIndexParams Index parameters for the output index
   */
  private CagraMergeParams(CagraIndexParams outputIndexParams) {
    this.outputIndexParams = outputIndexParams;
  }

  /**
   * Gets the index parameters for the output index.
   *
   * @return Index parameters to use for the output index
   */
  public CagraIndexParams getOutputIndexParams() {
    return outputIndexParams;
  }

  /**
   * Builder class for {@link CagraMergeParams}.
   */
  public static class Builder {
    private CagraIndexParams outputIndexParams = new CagraIndexParams.Builder().build();

    /**
     * Sets the index parameters for the output index.
     *
     * @param outputIndexParams Index parameters to use for the output index
     * @return This builder
     */
    public Builder withOutputIndexParams(CagraIndexParams outputIndexParams) {
      this.outputIndexParams = outputIndexParams;
      return this;
    }

    /**
     * Builds the {@link CagraMergeParams} object.
     *
     * @return The built parameters
     */
    public CagraMergeParams build() {
      return new CagraMergeParams(outputIndexParams);
    }
  }
}
