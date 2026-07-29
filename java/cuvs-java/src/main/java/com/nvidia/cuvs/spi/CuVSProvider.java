/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.spi;

import com.nvidia.cuvs.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodType;
import java.nio.file.Path;
import java.time.Duration;

/**
 * A provider of low-level cuvs resources and builders.
 */
public interface CuVSProvider {

  Path TMPDIR = Path.of(System.getProperty("java.io.tmpdir"));

  /**
   * The temporary directory to use for intermediate operations.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  static Path tempDirectory() {
    return TMPDIR;
  }

  /**
   * The directory where to extract and install the native library.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  default Path nativeLibraryPath() {
    return TMPDIR;
  }

  /** Creates a new CuVSResources. */
  CuVSResources newCuVSResources(Path tempDirectory) throws Throwable;

  /**
   * Creates a new CuVSResources whose memory allocations are tracked and
   * written as CSV samples from a background thread.
   *
   * <p>This method is declared as a {@code default} method so that adding it
   * does not break binary compatibility with providers compiled against an
   * earlier version of this interface; the default implementation throws
   * {@link UnsupportedOperationException} and providers must override it to
   * opt in.
   *
   * @param tempDirectory                the temporary directory to use for
   *                                     intermediate operations
   * @param memoryTrackingCsvPath        path to the output CSV file
   *                                     (created/truncated)
   * @param memoryTrackingSampleInterval minimum interval between successive
   *                                     CSV samples
   */
  default CuVSResources newCuVSResources(
      Path tempDirectory,
      Path memoryTrackingCsvPath,
      Duration memoryTrackingSampleInterval) throws Throwable {
    throw new UnsupportedOperationException(
        "Memory-tracking resources are not supported by this provider");
  }

  /** Create a {@link CuVSMatrix.Builder} instance for a host memory matrix **/
  CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long dimensions, CuVSMatrix.DataType dataType);

  /** Create a {@link CuVSMatrix.Builder} instance for a host memory matrix **/
  CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType);

  /** Create a {@link CuVSMatrix.Builder} instance for a device memory matrix **/
  CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources cuVSResources, long size, long dimensions, CuVSMatrix.DataType dataType);

  /** Create a {@link CuVSMatrix.Builder} instance for a device memory matrix **/
  CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources cuVSResources,
      long size,
      long dimensions,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType);

  /**
   * Returns the factory method used to build a CuVSMatrix from native memory.
   * The factory method will have this signature:
   * {@code CuVSMatrix createNativeMatrix(memorySegment, size, dimensions, dataType)},
   * where {@code memorySegment} is a {@code java.lang.foreign.MemorySegment} containing {@code int size} vectors of
   * {@code int dimensions} length of type {@link CuVSMatrix.DataType}.
   * <p>
   * In order to expose this factory in a way that is compatible with Java 21, the factory method is returned as a
   * {@link MethodHandle} with {@link MethodType} equal to
   * {@code (CuVSMatrix.class, MemorySegment.class, int.class, int.class, CuVSMatrix.DataType.class)}.
   * The caller will need to invoke the factory via the {@link MethodHandle#invokeExact} method:
   * {@code var matrix = (CuVSMatrix)newNativeMatrixBuilder().invokeExact(memorySegment, size, dimensions, dataType)}
   * </p>
   * @return a MethodHandle which can be invoked to build a CuVSMatrix from an external {@code MemorySegment}
   */
  MethodHandle newNativeMatrixBuilder();

  /**
   * Returns the factory method used to build a CuVSMatrix from native memory, with strides.
   * The factory method will have this signature:
   * {@code CuVSMatrix createNativeMatrix(memorySegment, size, dimensions, rowStride, columnStride, dataType)},
   * where {@code memorySegment} is a {@code java.lang.foreign.MemorySegment} containing {@code int size} vectors of
   * {@code int dimensions} length of type {@link CuVSMatrix.DataType}. Rows have a stride of {@code rowStride},
   * where 0 indicates "no stride" (a stride equal to the number of columns), and columns have a stride of
   * {@code columnStride}
   * <p>
   * In order to expose this factory in a way that is compatible with Java 21, the factory method is returned as a
   * {@link MethodHandle} with {@link MethodType} equal to
   * {@code (CuVSMatrix.class, MemorySegment.class, int.class, int.class, int.class, int.class, DataType.class)}.
   * The caller will need to invoke the factory via the {@link MethodHandle#invokeExact} method:
   * {@code var matrix = (CuVSMatrix)newNativeMatrixBuilder().invokeExact(memorySegment, size, dimensions, rowStride, columnStride, dataType)}
   * </p>
   * @return a MethodHandle which can be invoked to build a CuVSMatrix from an external {@code MemorySegment}
   */
  MethodHandle newNativeMatrixBuilderWithStrides();

  /** Create a {@link CuVSMatrix} from an on-heap array **/
  CuVSMatrix newMatrixFromArray(float[][] vectors);

  /** Create a {@link CuVSMatrix} from an on-heap array **/
  CuVSMatrix newMatrixFromArray(int[][] vectors);

  /** Create a {@link CuVSMatrix} from an on-heap array **/
  CuVSMatrix newMatrixFromArray(byte[][] vectors);

  /** Creates a new BruteForceIndex Builder. */
  BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /** Creates a new CagraIndex Builder. */
  CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /** Creates a new HnswIndex Builder. */
  HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /**
   * Creates an HNSW index from an existing CAGRA index.
   *
   * @param hnswParams Parameters for the HNSW index
   * @param cagraIndex The CAGRA index to convert from
   * @return A new HNSW index
   * @throws Throwable if an error occurs during conversion
   */
  HnswIndex hnswIndexFromCagra(HnswIndexParams hnswParams, CagraIndex cagraIndex) throws Throwable;

  /**
   * Builds an HNSW index using the ACE (Augmented Core Extraction) algorithm.
   *
   * @param resources The CuVS resources
   * @param hnswParams Parameters for the HNSW index with ACE configuration
   * @param dataset The dataset to build the index from
   * @return A new HNSW index ready for search
   * @throws Throwable if an error occurs during building
   */
  HnswIndex hnswIndexBuild(CuVSResources resources, HnswIndexParams hnswParams, CuVSMatrix dataset)
      throws Throwable;

  /** Creates a new TieredIndex Builder. */
  TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /**
   * Merges multiple CAGRA indexes into a single index.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  CagraIndex mergeCagraIndexes(CagraIndex[] indexes) throws Throwable;

  /**
   * Merges multiple CAGRA indexes into a single index with the specified merge parameters.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  default CagraIndex mergeCagraIndexes(CagraIndex[] indexes, CagraIndexParams mergeParams)
      throws Throwable {
    // Default implementation falls back to the method without parameters
    return mergeCagraIndexes(indexes);
  }

  /** Returns a {@link GPUInfoProvider} to query the system for GPU related information */
  GPUInfoProvider gpuInfoProvider();

  void setLogLevel(java.util.logging.Level level);

  java.util.logging.Level getLogLevel();

  /**
   * Switch RMM allocations (used internally by various cuVS algorithms and by the default implementation of
   * {@link CuVSDeviceMatrix}) to use pooled memory.
   * This operation has a global effect, and will affect all resources on the current device.
   *
   * @param initialPoolSizePercent The initial pool size, in percentage of the total GPU memory
   * @param maxPoolSizePercent The maximum pool size, in percentage of the total GPU memory
   */
  void enableRMMPooledMemory(int initialPoolSizePercent, int maxPoolSizePercent);

  /**
   * Switch RMM allocations (used internally by various cuVS algorithms and by the default implementation of
   * {@link CuVSDeviceMatrix}) to use pooled memory.
   * This operation has a global effect, and will affect all resources on the current device.
   *
   * @param initialPoolSizePercent The initial pool size, in percentage of the total GPU memory
   * @param maxPoolSizePercent The maximum pool size, in percentage of the total GPU memory
   */
  void enableRMMManagedPooledMemory(int initialPoolSizePercent, int maxPoolSizePercent);

  /** Disables pooled memory on the current device, reverting back to the default setting.  */
  void resetRMMPooledMemory();

  /** Retrieves the system-wide provider. */
  static CuVSProvider provider() {
    return CuVSServiceProvider.Holder.INSTANCE;
  }

  /**
   * Create a CAGRA index parameters compatible with HNSW index
   *
   * Note: The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce
   * exactly the same recalls and QPS for the same parameter `ef`. The graphs are different
   * internally. Depending on the selected heuristics, the CAGRA-produced graph's QPS-Recall curve
   * may be shifted along the curve right or left. See the heuristics descriptions for more details.
   *
   * @param rows The number of rows in the input dataset
   * @param dim The number of dimensions in the input dataset
   * @param m HNSW index parameter M
   * @param efConstruction HNSW index parameter ef_construction
   * @param heuristic The heuristic to use for selecting the graph build parameters
   * @param metric The distance metric to search
   * @return A new CAGRA index parameters object
   */
  CagraIndexParams cagraIndexParamsFromHnswParams(
      long rows,
      long dim,
      int m,
      int efConstruction,
      CagraIndexParams.HnswHeuristicType heuristic,
      CagraIndexParams.CuvsDistanceType metric);

  /**
   * Create CAGRA index parameters heuristically tuned for a dataset.
   *
   * @param rows The number of rows in the input dataset
   * @param dim The number of dimensions in the input dataset
   * @param graphDegree Degree of the output graph
   * @param metric The distance metric to search
   * @param buildQuality Higher values increase build quality (and cost) up to a point
   * @return A new CAGRA index parameters object
   */
  CagraIndexParams cagraIndexParamsFromDataset(
      long rows,
      long dim,
      long graphDegree,
      CagraIndexParams.CuvsDistanceType metric,
      long buildQuality);
}
