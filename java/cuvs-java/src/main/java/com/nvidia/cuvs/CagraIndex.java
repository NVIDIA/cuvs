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
  /** Caller-owned non-owning dataset view handle. */
  abstract class DatasetView implements AutoCloseable {
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

  /** Caller-owned padded dataset view. */
  final class PaddedDatasetView extends DatasetView {
    public PaddedDatasetView() {}
  }

  /** Caller-owned standard dataset view. */
  final class StandardDatasetView extends DatasetView {
    public StandardDatasetView() {}
  }

  /**
   * Caller-owned dataset handle populated by explicit deserialization or created by
   * {@link #makePaddedDataset(CuVSMatrix)}.
   */
  abstract class DeserializeDataset implements AutoCloseable {
    private AutoCloseable delegate;
    private long handleAddress;

    /**
     * Internal wiring hook used by the Java wrapper implementation.
     */
    public final void setDelegate(AutoCloseable delegate) {
      setDelegate(delegate, 0);
    }

    /**
     * Internal wiring hook used by the Java wrapper implementation.
     */
    public final void setDelegate(AutoCloseable delegate, long handleAddress) {
      this.delegate = delegate;
      this.handleAddress = handleAddress;
    }

    /**
     * Returns true when this handle owns native dataset storage.
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

  /**
   * Owning padded dataset handle. Keep this alive for as long as any index using it remains in
   * use.
   */
  final class PaddedDataset extends DeserializeDataset {
    public PaddedDataset() {}
  }

  /** Owning standard dataset handle populated by deserialization. */
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
   * Create an owning padded dataset by allocating padded storage and copying
   * {@code dataset}. Prefer this when the source matrix is not already padded to CAGRA's
   * required row stride (e.g. unaligned dimensions).
   */
  PaddedDataset makePaddedDataset(CuVSMatrix dataset) throws Throwable;

  /**
   * Create a caller-owned padded dataset view handle from a matrix that is already
   * padded to CAGRA's required row stride. For unpadded matrices use
   * {@link #makePaddedDataset(CuVSMatrix)}.
   */
  PaddedDatasetView makePaddedDatasetView(CuVSMatrix dataset) throws Throwable;

  /** Create a caller-owned standard dataset view handle from a matrix. */
  StandardDatasetView makeStandardDatasetView(CuVSMatrix dataset) throws Throwable;

  /**
   * Update this index with a caller-provided padded device dataset view and leave it
   * search-ready in padded-device layout. The caller retains ownership of the underlying
   * padded storage and must keep it alive while this index uses it.
   */
  void updateDataset(PaddedDatasetView datasetView) throws Throwable;

  /**
   * Update this index with a caller-owned padded device dataset. The dataset must remain alive
   * while this index uses it.
   */
  void updateDataset(PaddedDataset dataset) throws Throwable;

  /** Returns the CAGRA graph
   *
   * @return a {@link CuVSDeviceMatrix} encapsulating the native int (uint32_t) array used to represent
   * the cagra graph
   */
  CuVSDeviceMatrix getGraph();

  /**
   * Returns the degree of the built CAGRA graph (its number of edges per node), which may be
   * smaller than the requested {@code graph_degree} when the dataset is small enough that the
   * build truncated it.
   *
   * @return the built graph degree ({@code graph().extent(1)})
   */
  long getGraphDegree();

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
   * <p>The caller is responsible for concatenating every input index's dataset (in {@code
   * indexes} order) into a single caller-owned padded dataset before calling this, and for
   * providing {@code offsets}: entry {@code i} is the row at which {@code indexes[i]}'s rows
   * start in {@code mergedDataset}, and the last entry ({@code offsets[indexes.length]}) must
   * equal {@code mergedDataset}'s total row count. For example, with no filtering, {@code
   * offsets} is simply the cumulative row counts of {@code indexes} in order. This mirrors the
   * caller-owned-buffer contract used by {@link #updateDataset(PaddedDataset)}. Keep {@code
   * mergedDataset} alive for as long as the returned index remains in use.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergedDataset Caller-owned padded dataset holding the concatenation of every input
   *                      index's rows, in {@code indexes} order
   * @param offsets Per-index starting row within {@code mergedDataset}. Array of {@code
   *                indexes.length + 1} entries; the last entry must equal {@code mergedDataset}'s
   *                row count
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  static CagraIndex merge(CagraIndex[] indexes, PaddedDataset mergedDataset, long[] offsets)
      throws Throwable {
    return merge(indexes, mergedDataset, offsets, null);
  }

  /**
   * Merges multiple CAGRA indexes into a single index with the specified merge parameters.
   *
   * <p>See {@link #merge(CagraIndex[], PaddedDataset, long[])} for the {@code mergedDataset}/
   * {@code offsets} contract.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergedDataset Caller-owned padded dataset holding the concatenation of every input
   *                      index's rows, in {@code indexes} order
   * @param offsets Per-index starting row within {@code mergedDataset}. Array of {@code
   *                indexes.length + 1} entries; the last entry must equal {@code mergedDataset}'s
   *                row count
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  static CagraIndex merge(
      CagraIndex[] indexes,
      PaddedDataset mergedDataset,
      long[] offsets,
      CagraIndexParams mergeParams)
      throws Throwable {
    validateMergeArgs(indexes, offsets);
    Objects.requireNonNull(mergedDataset);
    if (!mergedDataset.isPresent()) {
      throw new IllegalArgumentException("mergedDataset is uninitialized");
    }
    return CuVSProvider.provider()
        .mergeCagraIndexes(indexes, mergedDataset.nativeHandleAddress(), offsets, mergeParams);
  }

  /**
   * Merges multiple CAGRA indexes into a single index using default merge parameters, from a
   * caller-owned padded dataset view over a buffer that is already padded to CAGRA's required
   * row stride.
   *
   * <p>See {@link #merge(CagraIndex[], PaddedDataset, long[])} for the {@code mergedDataset}/
   * {@code offsets} contract.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergedDataset Caller-owned padded dataset view holding the concatenation of every
   *                      input index's rows, in {@code indexes} order
   * @param offsets Per-index starting row within {@code mergedDataset}. Array of {@code
   *                indexes.length + 1} entries; the last entry must equal {@code mergedDataset}'s
   *                row count
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  static CagraIndex merge(CagraIndex[] indexes, PaddedDatasetView mergedDataset, long[] offsets)
      throws Throwable {
    return merge(indexes, mergedDataset, offsets, null);
  }

  /**
   * Merges multiple CAGRA indexes into a single index with the specified merge parameters, from a
   * caller-owned padded dataset view over a buffer that is already padded to CAGRA's required
   * row stride.
   *
   * <p>See {@link #merge(CagraIndex[], PaddedDataset, long[])} for the {@code mergedDataset}/
   * {@code offsets} contract.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergedDataset Caller-owned padded dataset view holding the concatenation of every
   *                      input index's rows, in {@code indexes} order
   * @param offsets Per-index starting row within {@code mergedDataset}. Array of {@code
   *                indexes.length + 1} entries; the last entry must equal {@code mergedDataset}'s
   *                row count
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  static CagraIndex merge(
      CagraIndex[] indexes,
      PaddedDatasetView mergedDataset,
      long[] offsets,
      CagraIndexParams mergeParams)
      throws Throwable {
    validateMergeArgs(indexes, offsets);
    Objects.requireNonNull(mergedDataset);
    if (!mergedDataset.isPresent()) {
      throw new IllegalArgumentException("mergedDataset is uninitialized");
    }
    return CuVSProvider.provider()
        .mergeCagraIndexes(indexes, mergedDataset.nativeHandleAddress(), offsets, mergeParams);
  }

  private static void validateMergeArgs(CagraIndex[] indexes, long[] offsets) {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }
    Objects.requireNonNull(offsets);
    if (offsets.length != indexes.length + 1) {
      throw new IllegalArgumentException(
          "offsets must have indexes.length + 1 entries, got " + offsets.length);
    }

    CuVSResources resources = indexes[0].getCuVSResources();
    for (int i = 1; i < indexes.length; i++) {
      if (!resources.equals(indexes[i].getCuVSResources())) {
        throw new IllegalArgumentException("All indexes must use the same CuVSResources instance");
      }
    }
  }

  /**
   * Reports whether the rows of {@code dataset} already sit at the row stride CAGRA requires, which
   * is the row length in bytes rounded up to a 16 byte boundary.
   *
   * <p>Use it to pick between the two padded dataset factories: a matrix that is already padded has
   * to go through {@link #makePaddedDatasetView(CuVSMatrix)}, because cuVS rejects a request to
   * copy it into padded storage it already occupies, and one that is not has to go through
   * {@link #makePaddedDataset(CuVSMatrix)}.
   *
   * @param dataset the matrix to inspect
   * @return true when the rows are already padded the way CAGRA requires
   */
  static boolean isPaddedDataset(CuVSMatrix dataset) {
    Objects.requireNonNull(dataset);
    return CuVSProvider.provider().isCagraPaddedDataset(dataset);
  }

  /**
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  interface Builder {

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed.
     *
     * @param inputStream an instance of {@link InputStream}
     * @return an instance of this Builder
     */
    Builder from(InputStream inputStream);

    /**
     * Sets an input stream and an empty caller-owned output handle for explicit dataset
     * deserialization. The concrete output type must match the dataset layout stored in the
     * serialized index. Keep {@code outDataset} alive while the built index is in use.
     *
     * @param inputStream an instance of {@link InputStream}
     * @param outDataset an empty {@link PaddedDataset} or {@link StandardDataset}
     * @return an instance of this Builder
     */
    Builder from(InputStream inputStream, DeserializeDataset outDataset);

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
