/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.Objects;

/**
 * {@link CagraIndex} encapsulates a CAGRA index, along with methods to interact
 * with it.
 * <p>
 * CAGRA is a graph-based nearest neighbors algorithm that was built from the
 * ground up for GPU acceleration. CAGRA demonstrates state-of-the art index
 * build and query performance for both small and large-batch sized search. Know
 * more about this algorithm
 * <a href="https://arxiv.org/abs/2308.15136" target="_blank">here</a>
 *
 * @since 25.02
 */
public interface CagraIndex extends AutoCloseable {
  /**
   * Caller-owned input dataset view handle for host-index attach.
   */
  abstract class DeviceDatasetView implements AutoCloseable {
    private AutoCloseable delegate;
    private long handleAddress;

    /**
     * Internal wiring hook used by the Java wrapper implementation.
     */
    public final void setDelegate(AutoCloseable delegate, long handleAddress) {
      this.delegate = delegate;
      this.handleAddress = handleAddress;
    }

    /**
     * Returns true when this view has a native handle.
     */
    public final boolean isPresent() {
      return delegate != null && handleAddress != 0;
    }

    /**
     * Internal accessor for native handle address.
     */
    public final long nativeHandleAddress() {
      return handleAddress;
    }

    @Override
    public void close() throws Exception {
      if (delegate != null) {
        delegate.close();
        delegate = null;
      }
      handleAddress = 0;
    }
  }

  /** Caller-owned device padded dataset view for host-index attach. */
  final class DevicePaddedDatasetView extends DeviceDatasetView {
    public DevicePaddedDatasetView() {}
  }

  /** Caller-owned device standard dataset view for host-index attach. */
  final class DeviceStandardDatasetView extends DeviceDatasetView {
    public DeviceStandardDatasetView() {}
  }

  /**
   * Caller-owned output dataset handle populated by {@link #deserialize(InputStream, DeserializeDataset)}.
   */
  abstract class DeserializeDataset implements AutoCloseable {
    private AutoCloseable delegate;

    /**
     * Internal wiring hook used by the Java wrapper implementation.
     */
    public final void setDelegate(AutoCloseable delegate) {
      this.delegate = delegate;
    }

    /**
     * Returns true when deserialization populated this handle with dataset payload ownership.
     */
    public final boolean isPresent() {
      return delegate != null;
    }

    @Override
    public void close() throws Exception {
      if (delegate != null) {
        delegate.close();
        delegate = null;
      }
    }
  }

  /** Output dataset handle for padded deserialize path. */
  final class PaddedDataset extends DeserializeDataset {
    public PaddedDataset() {}
  }

  /** Output dataset handle for standard deserialize path. */
  final class StandardDataset extends DeserializeDataset {
    public StandardDataset() {}
  }

  /**
   * Invokes the native destroy_cagra_index to de-allocate the CAGRA index
   */
  @Override
  void close() throws Exception;

  /**
   * Invokes the native search_cagra_index via the Panama API for searching a
   * CAGRA index.
   *
   * @param query an instance of {@link CagraQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link SearchResults} containing the results
   */
  SearchResults search(CagraQuery query) throws Throwable;

  /**
   * Create a caller-owned device padded dataset view handle from a device matrix.
   */
  DevicePaddedDatasetView makeDevicePaddedDatasetView(CuVSDeviceMatrix dataset) throws Throwable;

  /**
   * Create a caller-owned device standard dataset view handle from a device matrix.
   */
  DeviceStandardDatasetView makeDeviceStandardDatasetView(CuVSDeviceMatrix dataset)
      throws Throwable;

  /**
   * Convert a host-built index to device by attaching a caller-provided padded device dataset view.
   */
  void attachDevicePaddedDatasetOnHostIndex(DevicePaddedDatasetView datasetView) throws Throwable;

  /**
   * Convert a host-built index to device by attaching a caller-provided standard device dataset view.
   */
  void attachDeviceStandardDatasetOnHostIndex(DeviceStandardDatasetView datasetView)
      throws Throwable;

  /**
   * Deserializes into this pre-allocated index and optionally populates an output dataset handle.
   * <p>
   * Pass a {@link PaddedDataset} or {@link StandardDataset} to receive ownership of deserialized
   * dataset payload. Passing {@code null} mirrors the C/C++ optional out-dataset contract; this is
   * valid only when serialized data does not contain dataset payload.
   */
  void deserialize(InputStream inputStream, DeserializeDataset outDataset) throws Throwable;

  /** Returns the CAGRA graph
   *
   * @return a {@link CuVSDeviceMatrix} encapsulating the native int (uint32_t) array used to represent
   * the cagra graph
   */
  CuVSDeviceMatrix getGraph();

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * for writing index bytes.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes into
   */
  void serialize(OutputStream outputStream) throws Throwable;

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * for writing index bytes.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes into
   * @param bufferLength the length of buffer to use for writing bytes. Default
   *                     value is 1024
   */
  void serialize(OutputStream outputStream, int bufferLength) throws Throwable;

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * for writing index bytes.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes into
   * @param tempFile     an intermediate {@link Path} where CAGRA index is written
   *                     temporarily
   */
  default void serialize(OutputStream outputStream, Path tempFile) throws Throwable {
    serialize(outputStream, tempFile, 1024);
  }

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * and path to the intermediate temporary file.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param tempFile     an intermediate {@link Path} where CAGRA index is written
   *                     temporarily
   * @param bufferLength the length of buffer to use for writing bytes. Default
   *                     value is 1024
   */
  void serialize(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable;

  /**
   * A method to create and persist HNSW index from CAGRA index using an instance
   * of {@link OutputStream} and path to the intermediate temporary file.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   */
  void serializeToHNSW(OutputStream outputStream) throws Throwable;

  /**
   * A method to create and persist HNSW index from CAGRA index using an instance
   * of {@link OutputStream} and path to the intermediate temporary file.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param bufferLength the length of buffer to use for writing bytes. Default
   *                     value is 1024
   */
  void serializeToHNSW(OutputStream outputStream, int bufferLength) throws Throwable;

  /**
   * A method to create and persist HNSW index from CAGRA index using an instance
   * of {@link OutputStream} and path to the intermediate temporary file.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param tempFile     an intermediate {@link Path} where CAGRA index is written
   *                     temporarily
   */
  default void serializeToHNSW(OutputStream outputStream, Path tempFile) throws Throwable {
    serializeToHNSW(outputStream, tempFile, 1024);
  }

  /**
   * A method to create and persist HNSW index from CAGRA index using an instance
   * of {@link OutputStream} and path to the intermediate temporary file.
   *
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param tempFile     an intermediate {@link Path} where CAGRA index is written
   *                     temporarily
   * @param bufferLength the length of buffer to use for writing bytes. Default
   *                     value is 1024
   */
  void serializeToHNSW(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable;

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
   * @throws UnsupportedOperationException if the provider does not cuvs
   */
  static Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    return CuVSProvider.provider().newCagraIndexBuilder(cuvsResources);
  }

  /**
   * Merges multiple CAGRA indexes into a single index using default merge parameters.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  static CagraIndex merge(CagraIndex[] indexes) throws Throwable {
    return merge(indexes, null);
  }

  /**
   * Merges multiple CAGRA indexes into a single index with the specified merge parameters.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  static CagraIndex merge(CagraIndex[] indexes, CagraIndexParams mergeParams) throws Throwable {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }

    CuVSResources resources = indexes[0].getCuVSResources();
    for (int i = 1; i < indexes.length; i++) {
      if (!resources.equals(indexes[i].getCuVSResources())) {
        throw new IllegalArgumentException("All indexes must use the same CuVSResources instance");
      }
    }

    return CuVSProvider.provider().mergeCagraIndexes(indexes, mergeParams);
  }

  /**
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  interface Builder {

    /**
     * Sets a CAGRA graph instance to re-create an index from a
     * previously built graph.
     */
    Builder from(CuVSMatrix graph);

    /**
     * Sets the dataset vectors for building the {@link CagraIndex}.
     *
     * @param vectors a two-dimensional float array
     * @return an instance of this Builder
     */
    Builder withDataset(float[][] vectors);

    /**
     * Sets the dataset for building the {@link CagraIndex}.
     *
     * @param dataset a {@link CuVSMatrix} object containing the vectors
     * @return an instance of this Builder
     */
    Builder withDataset(CuVSMatrix dataset);

    /**
     * Registers an instance of configured {@link CagraIndexParams} with this
     * Builder.
     *
     * @param cagraIndexParameters An instance of CagraIndexParams.
     * @return An instance of this Builder.
     */
    Builder withIndexParams(CagraIndexParams cagraIndexParameters);

    /**
     * Builds and returns an instance of CagraIndex.
     *
     * @return an instance of CagraIndex
     */
    CagraIndex build() throws Throwable;
  }
}
