/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.BitSet;
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
   * Invokes the native destroy_cagra_index to de-allocate the CAGRA index. Also attempts to close
   * any dataset whose ownership transferred to this index during construction.
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
   * Returns the number of vectors in this index.
   *
   * @return the number of rows of the indexed dataset
   */
  long size();

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
   * Creates a new Builder with an instance of {@link CuVSResources}. Pick what the index is created
   * from with one of the {@code from*} methods of {@link Builder}, then call {@code build()} on the
   * builder it returns.
   *
   * @param cuvsResources an instance of {@link CuVSResources}
   * @throws UnsupportedOperationException if cuVS is not available, for example without a GPU or
   *     without the native library
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
    return merge(indexes, null, null);
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
    return merge(indexes, mergeParams, null);
  }

  /**
   * Merges multiple CAGRA indexes into a single index, keeping only the rows selected by
   * {@code rowFilter}.
   *
   * <p>The merge concatenates the input datasets in the order the indexes are given, so bit
   * {@code i} of the filter refers to row {@code i} of that concatenation: bits {@code 0} to
   * {@code indexes[0].size() - 1} address the first index, the bits that follow address the second,
   * and so on. A <b>set</b> bit keeps the row; a clear bit drops it. The rows that survive keep
   * their relative order and are packed together, so the merged index has one row per set bit.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @param rowFilter The rows to keep, or null to keep all of them. A BitSet shorter than the total
   *     row count is valid: the rows beyond its logical length are treated as clear (dropped). A bit
   *     set at a position at or beyond the total row count throws {@link IllegalArgumentException}.
   * @return A new merged CAGRA index
   * @throws IllegalArgumentException if {@code rowFilter} has a bit set beyond the last row, or if
   *     it is non-null but keeps no rows at all
   * @throws Throwable if an error occurs during the merge operation
   */
  static CagraIndex merge(CagraIndex[] indexes, CagraIndexParams mergeParams, BitSet rowFilter)
      throws Throwable {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }

    CuVSResources resources = indexes[0].getCuVSResources();
    for (int i = 1; i < indexes.length; i++) {
      if (!resources.equals(indexes[i].getCuVSResources())) {
        throw new IllegalArgumentException("All indexes must use the same CuVSResources instance");
      }
    }

    return CuVSProvider.provider().mergeCagraIndexes(indexes, mergeParams, rowFilter);
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
   * Builder helps configure and create an instance of {@link CagraIndex} from a variety of sources.
   *
   * <p>Call one of the {@code from*} methods on the builder returned by
   * {@link CagraIndex#newBuilder(CuVSResources)}. Each one takes the inputs that kind of index
   * requires and returns a builder for the optional ones, whose {@code build()} creates the index:
   *
   * <pre>{@code
   * CagraIndex index =
   *     CagraIndex.newBuilder(resources).fromDataset(dataset).withIndexParams(params).build();
   * CagraIndex loaded = CagraIndex.newBuilder(resources).fromSerialized(inputStream).build();
   * }</pre>
   *
   * <p>A builder creates one index: a second {@code from*} call, or a {@code from*} call combined
   * with the deprecated setters, throws {@link IllegalStateException}. The builders the
   * {@code from*} methods return take each input once.
   *
   * <p>Each {@link com.nvidia.cuvs.spi.CuVSProvider} returns its own implementation of this
   * interface from {@link com.nvidia.cuvs.spi.CuVSProvider#newCagraIndexBuilder(CuVSResources)}.
   * The {@code from*} methods are {@code default} methods that throw
   * {@link UnsupportedOperationException}, so that implementations written before they existed
   * still compile; the built-in provider implements all of them.
   */
  interface Builder {

    /**
     * Builds the CAGRA graph from dense vectors.
     *
     * <p>The index takes ownership of {@code dataset} when {@link FromDatasetBuilder#build()}
     * returns, and closes it when the index is closed. If {@code build()} throws, the caller still
     * owns it.
     *
     * @param dataset the vectors to index
     * @return a builder for the optional inputs
     */
    default FromDatasetBuilder fromDataset(CuVSMatrix dataset) {
      throw notImplemented("fromDataset");
    }

    /**
     * Builds the CAGRA graph from dense vectors held in a Java array.
     * {@link FromDatasetBuilder#build()} checks them and copies them into a matrix that the index
     * owns, so changes made to the array before then reach the index.
     *
     * @param vectors the vectors to index, one row per vector, all of the same length
     * @return a builder for the optional inputs
     */
    default FromDatasetBuilder fromDataset(float[][] vectors) {
      throw notImplemented("fromDataset");
    }

    /**
     * Builds the CAGRA graph from one BBQ encoding of the vectors, used on both sides of every
     * distance. Only nn-descent graph construction is supported.
     *
     * <p>The index stores views over the quantizer's matrices rather than copying them, so they
     * must stay open for as long as the index is in use. Unless a dense dataset is attached with
     * {@link FromBbqBuilder#withDenseDataset(CuVSMatrix)}, the index can't be searched until
     * {@link CagraIndex#updateDataset(PaddedDatasetView)} or
     * {@link CagraIndex#updateDataset(PaddedDataset)} attaches one.
     *
     * @param quantizer the encoded vectors
     * @return a builder for the optional inputs
     */
    default FromBbqBuilder fromBbq(BbqQuantizer quantizer) {
      throw notImplemented("fromBbq");
    }

    /**
     * Builds the CAGRA graph from two BBQ encodings of the same vectors at different precisions.
     * They can be given in either order: cuVS uses the lower-precision one for the stored side of
     * every distance and the other one for the query side. The supported pairs of layouts are
     * {@code PACKED_1B} with {@code PACKED_4B}, {@code TRANSPOSED_2B} or {@code TRANSPOSED_4B}, and
     * {@code TRANSPOSED_2B} with {@code TRANSPOSED_4B}.
     *
     * <p>Otherwise this behaves like {@link #fromBbq(BbqQuantizer)}, and both quantizers must stay
     * open for as long as the index is in use.
     *
     * @param quantizer one encoding of the vectors
     * @param other the other encoding, with a different layout
     * @return a builder for the optional inputs
     * @throws IllegalArgumentException if both quantizers use the same layout
     */
    default FromBbqBuilder fromBbq(BbqQuantizer quantizer, BbqQuantizer other) {
      throw notImplemented("fromBbq");
    }

    /**
     * Creates an index around a graph built earlier and the dataset it was built from. No graph is
     * built.
     *
     * <p>The index takes ownership of {@code dataset} when {@link FromGraphBuilder#build()}
     * returns. It never owns {@code graph}: a graph in host memory is copied, but one in device
     * memory is used in place, so it must stay open for as long as the index is in use. The graph
     * returned by {@link CagraIndex#getGraph()} is a view into its index's own device memory, so
     * passing it directly ties the new index to that one, which must then stay open too. Pass a
     * copy made with {@link CuVSMatrix#toHost()} to keep the two independent.
     *
     * @param metric the distance the graph was built for
     * @param graph the graph, one row of neighbor indices per vector
     * @param dataset the vectors the graph was built from, in device memory
     * @return a builder for the optional inputs
     * @throws IllegalArgumentException if {@code dataset} is not in device memory
     */
    default FromGraphBuilder fromGraph(
        CagraIndexParams.CuvsDistanceType metric, CuVSMatrix graph, CuVSMatrix dataset) {
      throw notImplemented("fromGraph");
    }

    /**
     * Loads an index written by {@link CagraIndex#serialize(OutputStream)}, with its graph and
     * dataset. {@link FromSerializedBuilder#build()} reads {@code inputStream} to its end, but
     * closing it is left to the caller. The index owns the loaded dataset unless
     * {@link FromSerializedBuilder#withOutputDataset(DeserializeDataset)} hands it to the caller.
     *
     * @param inputStream the serialized index
     * @return a builder for the optional inputs
     */
    default FromSerializedBuilder fromSerialized(InputStream inputStream) {
      throw notImplemented("fromSerialized");
    }

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed. Unlike {@link #fromSerialized(InputStream)}, {@link #build()} closes it.
     *
     * @param inputStream an instance of {@link InputStream}
     * @return an instance of this Builder
     * @deprecated Use {@link #fromSerialized(InputStream)}.
     */
    @Deprecated(since = "26.12", forRemoval = true)
    Builder from(InputStream inputStream);

    /**
     * Sets an input stream and an empty caller-owned output handle for explicit dataset
     * deserialization. The concrete output type must match the dataset layout stored in the
     * serialized index. Keep {@code outDataset} alive while the built index is in use. Unlike
     * {@link #fromSerialized(InputStream)}, {@link #build()} closes {@code inputStream}.
     *
     * @param inputStream an instance of {@link InputStream}
     * @param outDataset an empty {@link PaddedDataset} or {@link StandardDataset}
     * @return an instance of this Builder
     * @deprecated Use {@link #fromSerialized(InputStream)} and
     *     {@link FromSerializedBuilder#withOutputDataset(DeserializeDataset)}.
     */
    @Deprecated(since = "26.12", forRemoval = true)
    Builder from(InputStream inputStream, DeserializeDataset outDataset);

    /**
     * Sets a CAGRA graph instance to re-create an index from a
     * previously built graph.
     *
     * @deprecated Use {@link #fromGraph(CagraIndexParams.CuvsDistanceType, CuVSMatrix,
     *     CuVSMatrix)}, which takes the metric directly.
     */
    @Deprecated(since = "26.12", forRemoval = true)
    Builder from(CuVSMatrix graph);

    /**
     * Sets the dataset vectors for building the {@link CagraIndex}.
     *
     * @param vectors a two-dimensional float array
     * @return an instance of this Builder
     * @deprecated Use {@link #fromDataset(float[][])}.
     */
    @Deprecated(since = "26.12", forRemoval = true)
    Builder withDataset(float[][] vectors);

    /**
     * Sets the dataset for building the {@link CagraIndex}.
     *
     * <p>The caller retains ownership until a build that uses this dataset returns successfully.
     * The returned index then owns the dataset, and the caller must leave it open until the index
     * is closed. If the build fails or uses another configured input source, ownership remains
     * with the caller.
     *
     * @param dataset a {@link CuVSMatrix} object containing the vectors
     * @return an instance of this Builder
     * @deprecated Use {@link #fromDataset(CuVSMatrix)}. To create an index from a graph or from
     *     BBQ quantizers, pass the dataset to {@link #fromGraph(CagraIndexParams.CuvsDistanceType,
     *     CuVSMatrix, CuVSMatrix)} or {@link FromBbqBuilder#withDenseDataset(CuVSMatrix)}.
     */
    @Deprecated(since = "26.12", forRemoval = true)
    Builder withDataset(CuVSMatrix dataset);

    /**
     * Builds the graph from one or two encoded BBQ representations. An optional dense dataset
     * supplied with {@link #withDataset(CuVSMatrix)} is attached before search; otherwise call
     * {@link CagraIndex#updateDataset(PaddedDatasetView)} or
     * {@link CagraIndex#updateDataset(PaddedDataset)} before searching.
     *
     * <p>The index stores views over the quantizer tensors rather than copying them, so they must
     * stay open for as long as the index is in use. A dense dataset passed to
     * {@link #withDataset(CuVSMatrix)} is owned by the index, as it is for a non-BBQ build, and is
     * closed with it.
     *
     * @deprecated Use {@link #fromBbq(BbqQuantizer)} or {@link #fromBbq(BbqQuantizer,
     *     BbqQuantizer)}.
     */
    @Deprecated(since = "26.12", forRemoval = true)
    Builder withBbqDataset(BbqQuantizer... quantizers);

    /**
     * Registers an instance of configured {@link CagraIndexParams} with this
     * Builder.
     *
     * @param cagraIndexParameters An instance of CagraIndexParams.
     * @return An instance of this Builder.
     * @deprecated Use {@link FromDatasetBuilder#withIndexParams(CagraIndexParams)} or
     *     {@link FromBbqBuilder#withIndexParams(CagraIndexParams)}. An index created from a graph
     *     takes its metric as an argument of {@link #fromGraph(CagraIndexParams.CuvsDistanceType,
     *     CuVSMatrix, CuVSMatrix)}.
     */
    @Deprecated(since = "26.12", forRemoval = true)
    Builder withIndexParams(CagraIndexParams cagraIndexParameters);

    /**
     * Builds and returns an instance of CagraIndex. With a dataset alone, {@link #withIndexParams}
     * is optional, and without it the index is built with the defaults of
     * {@link CagraIndexParams.Builder}. With BBQ quantizers it is optional too, and without it the
     * index is built with those defaults and the quantizers' metric. With a graph it is required,
     * because the index takes its metric from it. With a stream to load it is not allowed,
     * because the loaded index has its own.
     *
     * @return an instance of CagraIndex
     * @deprecated Call {@code build()} on the builder returned by one of the {@code from*}
     *     methods.
     */
    @Deprecated(since = "26.12", forRemoval = true)
    CagraIndex build() throws Throwable;

    private UnsupportedOperationException notImplemented(String method) {
      return new UnsupportedOperationException(
          getClass().getName() + " does not implement CagraIndex.Builder." + method);
    }
  }

  /**
   * Builds a {@link CagraIndex} from dense vectors. Returned by
   * {@link Builder#fromDataset(CuVSMatrix)} and {@link Builder#fromDataset(float[][])}.
   */
  interface FromDatasetBuilder {

    /**
     * Sets the build parameters. Without this call the index is built with the defaults of
     * {@link CagraIndexParams.Builder}.
     *
     * @param indexParams the build parameters
     * @return this builder
     * @throws IllegalStateException if called more than once
     */
    FromDatasetBuilder withIndexParams(CagraIndexParams indexParams);

    /**
     * Builds the index. A builder creates one index, so this can be called only once,
     * even if it fails. To try again, start a new builder with
     * {@link CagraIndex#newBuilder(CuVSResources)}.
     *
     * @return the new index
     * @throws IllegalStateException if {@code build()} was already called, whether or not it
     *     succeeded
     * @throws IllegalArgumentException if the array given to {@link Builder#fromDataset(float[][])}
     *     is empty or its rows differ in length
     * @throws Throwable if the build fails
     */
    CagraIndex build() throws Throwable;
  }

  /**
   * Builds a {@link CagraIndex} from BBQ-encoded vectors. Returned by
   * {@link Builder#fromBbq(BbqQuantizer)} and {@link Builder#fromBbq(BbqQuantizer, BbqQuantizer)}.
   */
  interface FromBbqBuilder {

    /**
     * Attaches full-precision vectors once the graph is built, so that the index can be searched
     * straight away. The index takes ownership of {@code dataset} when {@link #build()} returns,
     * and closes it when the index is closed. If {@code build()} throws, the caller still owns it.
     *
     * @param dataset the full-precision vectors, in the same order as the encoded ones
     * @return this builder
     * @throws IllegalStateException if called more than once
     */
    FromBbqBuilder withDenseDataset(CuVSMatrix dataset);

    /**
     * Sets the build parameters. Their graph build algorithm must be
     * {@link CagraIndexParams.CagraGraphBuildAlgo#NN_DESCENT}, or
     * {@link CagraIndexParams.CagraGraphBuildAlgo#AUTO_SELECT}, which picks nn-descent, and their
     * metric must be the one the quantizers were encoded for. Without this call the index is built
     * with the defaults of {@link CagraIndexParams.Builder} and the quantizers' metric.
     *
     * @param indexParams the build parameters
     * @return this builder
     * @throws IllegalArgumentException if the parameters ask for another graph build algorithm, or
     *     for a metric other than the quantizers'
     * @throws IllegalStateException if called more than once
     */
    FromBbqBuilder withIndexParams(CagraIndexParams indexParams);

    /**
     * Builds the index. A builder creates one index, so this can be called only once,
     * even if it fails. To try again, start a new builder with
     * {@link CagraIndex#newBuilder(CuVSResources)}.
     *
     * @return the new index
     * @throws IllegalStateException if {@code build()} was already called, whether or not it
     *     succeeded
     * @throws Throwable if the build fails
     */
    CagraIndex build() throws Throwable;
  }

  /**
   * Creates a {@link CagraIndex} around a graph built earlier. Returned by
   * {@link Builder#fromGraph(CagraIndexParams.CuvsDistanceType, CuVSMatrix, CuVSMatrix)}.
   */
  interface FromGraphBuilder {

    /**
     * Creates the index. A builder creates one index, so this can be called only once,
     * even if it fails. To try again, start a new builder with
     * {@link CagraIndex#newBuilder(CuVSResources)}.
     *
     * @return the new index
     * @throws IllegalStateException if {@code build()} was already called, whether or not it
     *     succeeded
     * @throws Throwable if the index can't be created
     */
    CagraIndex build() throws Throwable;
  }

  /**
   * Loads a serialized {@link CagraIndex}. Returned by {@link Builder#fromSerialized(InputStream)}.
   */
  interface FromSerializedBuilder {

    /**
     * Hands the loaded dataset to the caller instead of the index. {@code outDataset} must be
     * empty, and its type must match the layout of the dataset stored in the serialized index: a
     * {@link PaddedDataset} or a {@link StandardDataset}. The index uses the dataset in place, so
     * keep {@code outDataset} open for as long as the index is in use.
     *
     * @param outDataset an empty handle that receives the dataset
     * @return this builder
     * @throws IllegalStateException if called more than once
     */
    FromSerializedBuilder withOutputDataset(DeserializeDataset outDataset);

    /**
     * Loads the index. A builder creates one index, so this can be called only once,
     * even if it fails. To try again, start a new builder with
     * {@link CagraIndex#newBuilder(CuVSResources)}.
     *
     * @return the loaded index
     * @throws IllegalStateException if {@code build()} was already called, whether or not it
     *     succeeded
     * @throws Throwable if the index can't be loaded
     */
    CagraIndex build() throws Throwable;
  }
}
