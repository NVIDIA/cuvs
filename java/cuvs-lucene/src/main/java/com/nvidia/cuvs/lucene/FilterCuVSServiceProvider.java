/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.spi.CuVSProvider;
import com.nvidia.cuvs.spi.CuVSServiceProvider;

/**
 * A provider that creates instances of FilterCuVSProvider.
 *
 * @since 25.10
 */
public class FilterCuVSServiceProvider extends CuVSServiceProvider {

  /**
   * Initialize and return an CuVSProvider provided by this provider.
   */
  @Override
  public CuVSProvider get(CuVSProvider builtinProvider) {
    return new FilterCuVSProvider(builtinProvider);
  }
}
