/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.nio.file.Path;
import java.util.Objects;

/**
 * {@link VamanaIndex} encapsulates a Vamana index, along with methods to build
 * it on the GPU and serialize it in the DiskANN file format.
 * <p>
 * Vamana is the graph construction algorithm behind DiskANN. cuVS currently
 * provides build and serialize only. There is no Vamana search API, so a
 * serialized index is searched by loading it with DiskANN.
 *
 * @since 26.10
 */
public interface VamanaIndex extends AutoCloseable {

  @Override
  void close() throws Exception;

  /**
   * Gets the dimensionality of the vectors in this index.
   *
   * @return the number of dimensions
   */
  int getDimensions() throws Throwable;

  /**
   * Serializes the index in the DiskANN file format, including the dataset.
   * <p>
   * This writes <b>two</b> files, {@code filePrefix} holding the graph and
   * {@code filePrefix + ".data"} holding the dataset.
   *
   * @param filePrefix the prefix that output file names are derived from
   */
  default void serialize(Path filePrefix) throws Throwable {
    serialize(filePrefix, true);
  }

  /**
   * Serializes the index in the DiskANN file format.
   * <p>
   * When {@code includeDataset} is true this writes {@code filePrefix} holding
   * the graph and {@code filePrefix + ".data"} holding the dataset. When it is
   * false only {@code filePrefix} is written.
   * <p>
   * The argument is a prefix and not a complete file name, matching the native
   * {@code file_prefix} parameter.
   *
   * @param filePrefix     the prefix that output file names are derived from
   * @param includeDataset whether to write the dataset alongside the graph
   */
  void serialize(Path filePrefix, boolean includeDataset) throws Throwable;

  /**
   * Gets an instance of {@link CuVSResources}
   *
   * @return an instance of {@link CuVSResources}
   */
  CuVSResources getCuVSResources();

  /**
   * Creates a new Builder with an instance of {@link CuVSResources}.
   *
   * @param cuvsResources an instance of {@link CuVSResources}
   * @throws UnsupportedOperationException if the provider does not support cuvs
   */
  static Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    return CuVSProvider.provider().newVamanaIndexBuilder(cuvsResources);
  }

  /**
   * Builder helps configure and create an instance of {@link VamanaIndex}.
   */
  interface Builder {

    /**
     * Sets the dataset for building the {@link VamanaIndex}.
     *
     * @param vectors a two-dimensional float array
     * @return an instance of this Builder
     */
    Builder withDataset(float[][] vectors);

    /**
     * Sets the dataset for building the {@link VamanaIndex}.
     * <p>
     * The native builder accepts {@code float}, {@code half}, {@code uint8},
     * and {@code int8} datasets. Of those, {@link CuVSMatrix.DataType#FLOAT},
     * {@link CuVSMatrix.DataType#HALF}, and {@link CuVSMatrix.DataType#BYTE}
     * are reachable from Java today, where {@code BYTE} is unsigned.
     * {@code int8} has no corresponding {@code DataType}.
     * <p>
     * The native index may retain a non-owning device view of the dataset
     * rather than copying it, so the caller must keep this matrix open for at
     * least as long as the index and close it afterwards. A dataset supplied as
     * a {@code float[][]} is created and closed by the index instead.
     *
     * @param dataset a {@link CuVSMatrix} object containing the vectors
     * @return an instance of this Builder
     */
    Builder withDataset(CuVSMatrix dataset);

    /**
     * Registers an instance of configured {@link VamanaIndexParams} with this
     * Builder.
     *
     * @param vamanaIndexParameters An instance of VamanaIndexParams
     * @return An instance of this Builder
     */
    Builder withIndexParams(VamanaIndexParams vamanaIndexParameters);

    /**
     * Builds and returns an instance of {@link VamanaIndex}.
     *
     * @return an instance of {@link VamanaIndex}
     */
    VamanaIndex build() throws Throwable;
  }
}
