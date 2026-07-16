/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.spi;

import static com.nvidia.cuvs.internal.CuVSParamsHelper.*;
import static com.nvidia.cuvs.internal.common.Util.*;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.*;
import com.nvidia.cuvs.internal.common.CloseableHandle;
import com.nvidia.cuvs.internal.common.PinnedMemoryBuffer;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.cuvsCagraIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqIndexParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqParams;
import com.nvidia.cuvs.internal.panama.cuvsIvfPqSearchParams;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.util.logging.Level;

final class JDKProvider implements CuVSProvider {

  record CuVSVersion(short major, short minor, short patch) {
    private static final int MAX_VERSION_DISTANCE = 3;

    static CuVSVersion fromString(String versionString) {
      var tokens = versionString.split("\\.");
      final short major = parseToken(tokens, 0);
      final short minor = parseToken(tokens, 1);
      final short patch = parseToken(tokens, 2);
      if (major == 0 || minor == 0) {
        return null;
      }
      return new CuVSVersion(major, minor, patch);
    }

    private static short parseToken(String[] tokens, int index) {
      if (index < tokens.length) {
        try {
          return Short.parseShort(tokens[index]);
        } catch (NumberFormatException _) {
        }
      }
      return 0;
    }

    static boolean isCuVSMoreRecent(CuVSVersion mavenVersion, CuVSVersion cuvsVersion) {
      return (cuvsVersion.major == mavenVersion.major && cuvsVersion.minor >= mavenVersion.minor)
          || cuvsVersion.major > mavenVersion.major;
    }

    static boolean isCuVSWithinMaxRange(CuVSVersion mavenVersion, CuVSVersion cuvsVersion)
        throws ProviderInitializationException {
      var maxVersionFromSyspropString = System.getProperty("cuvs.max_version");

      final CuVSVersion maxVersion;
      if (maxVersionFromSyspropString != null) {
        maxVersion = fromString(maxVersionFromSyspropString);
        if (maxVersion == null) {
          throw new ProviderInitializationException(
              "System property 'cuvs.max_version' is not a valid cuVS version: "
                  + maxVersionFromSyspropString);
        }
      } else {
        maxVersion = addReleases(mavenVersion, MAX_VERSION_DISTANCE);
      }

      return cuvsVersion.major < maxVersion.major
          || (cuvsVersion.major == maxVersion.major && cuvsVersion.minor <= maxVersion.minor);
    }

    private static CuVSVersion addReleases(CuVSVersion currentVersion, int numberOfReleases) {
      short candidateMinor = (short) (currentVersion.minor + numberOfReleases * 2);
      short releaseMinor = (short) (candidateMinor % 12);
      short releaseMajor = (short) (currentVersion.major + (candidateMinor / 12));
      if (releaseMinor == 0) {
        releaseMinor = 12;
        releaseMajor -= 1;
      }

      return new CuVSVersion(releaseMajor, releaseMinor, (short) 0);
    }

    @Override
    public String toString() {
      return String.format(Locale.ROOT, "%02d.%02d.%d", major, minor, patch);
    }
  }

  private static final MethodHandle createNativeDataset$mh;
  private static final MethodHandle createNativeDatasetWithStrides$mh;

  static {
    try {
      var lookup = MethodHandles.lookup();
      createNativeDataset$mh =
          lookup.findStatic(
              JDKProvider.class,
              "createNativeDataset",
              MethodType.methodType(
                  CuVSMatrix.class,
                  MemorySegment.class,
                  int.class,
                  int.class,
                  CuVSMatrix.DataType.class));

      createNativeDatasetWithStrides$mh =
          lookup.findStatic(
              JDKProvider.class,
              "createNativeDatasetWithStrides",
              MethodType.methodType(
                  CuVSMatrix.class,
                  MemorySegment.class,
                  int.class,
                  int.class,
                  int.class,
                  int.class,
                  CuVSMatrix.DataType.class));
    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  private final cuvsRMMMemoryResourceReset cuvsRMMMemoryResourceResetInvoker =
      cuvsRMMMemoryResourceReset.makeInvoker();

  private final cuvsRMMAsyncMemoryResourceEnable cuvsRMMAsyncMemoryResourceEnableInvoker =
      cuvsRMMAsyncMemoryResourceEnable.makeInvoker();

  private final cuvsGetLogLevel GET_LOG_LEVEL_INVOKER = cuvsGetLogLevel.makeInvoker();

  private JDKProvider() {}

  static CuVSProvider create() throws ProviderInitializationException {
    NativeDependencyLoader.loadLibraries();

    try (var localArena = Arena.ofConfined()) {
      var majorPtr = localArena.allocate(uint16_t);
      var minorPtr = localArena.allocate(uint16_t);
      var patchPtr = localArena.allocate(uint16_t);
      checkCuVSError(cuvsVersionGet(majorPtr, minorPtr, patchPtr), "cuvsVersionGet");
      var major = majorPtr.get(uint16_t, 0);
      var minor = minorPtr.get(uint16_t, 0);
      var patch = patchPtr.get(uint16_t, 0);

      var mavenVersionString = readCuVSVersionFromManifest();
      checkCuVSVersionMatching(mavenVersionString, major, minor, patch);
    }
    return new JDKProvider();
  }

  static void checkCuVSVersionMatching(
      String mavenVersionString, short major, short minor, short patch)
      throws ProviderInitializationException {
    var mavenVersion = CuVSVersion.fromString(mavenVersionString);
    var cuvsVersion = new CuVSVersion(major, minor, patch);

    if (mavenVersion != null) {
      if (!CuVSVersion.isCuVSMoreRecent(mavenVersion, cuvsVersion)) {
        throw new ProviderInitializationException(
            String.format(
                Locale.ROOT,
                """
                Version mismatch: outdated libcuvs_c (libcuvs_c [%s], cuvs-java version [%s]).\
                 Please upgrade your libcuvs_c installation to match at lease the cuvs-java\
                 version.\
                """,
                cuvsVersion,
                mavenVersion));
      }
      if (!CuVSVersion.isCuVSWithinMaxRange(mavenVersion, cuvsVersion)) {
        throw new ProviderInitializationException(
            String.format(
                Locale.ROOT,
                "Version mismatch: unsupported libcuvs_c (libcuvs_c [%s], cuvs-java version [%s]). "
                    + "Please upgrade your software, or install a previous version of libcuvs_c.",
                cuvsVersion,
                mavenVersion));
      }
    }
  }

  /**
   * Read cuvs-java version from this Jar Manifest, or null if these are not available
   */
  private static String readCuVSVersionFromManifest() {
    try (var jarFile =
        new JarFile(
            JDKProvider.class.getProtectionDomain().getCodeSource().getLocation().getPath())) {
      Manifest manifest = jarFile.getManifest();
      return manifest.getMainAttributes().getValue("Implementation-Version");
    } catch (IOException e) {
      return null;
    }
  }

  private static CuVSMatrix createNativeDataset(
      MemorySegment memorySegment, int size, int dimensions, CuVSMatrix.DataType dataType) {
    return new CuVSHostMatrixImpl(memorySegment, size, dimensions, dataType);
  }

  private static CuVSMatrix createNativeDatasetWithStrides(
      MemorySegment memorySegment,
      int size,
      int dimensions,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType) {
    return new CuVSHostMatrixImpl(
        memorySegment, size, dimensions, rowStride, columnStride, dataType);
  }

  @Override
  public CuVSResources newCuVSResources(Path tempDirectory) {
    Objects.requireNonNull(tempDirectory);
    if (Files.notExists(tempDirectory)) {
      throw new IllegalArgumentException("does not exist:" + tempDirectory);
    }
    if (!Files.isDirectory(tempDirectory)) {
      throw new IllegalArgumentException("not a directory:" + tempDirectory);
    }
    return new CuVSResourcesImpl(tempDirectory);
  }

  @Override
  public CuVSResources newCuVSResources(
      Path tempDirectory,
      Path memoryTrackingCsvPath,
      Duration memoryTrackingSampleInterval) {
    Objects.requireNonNull(tempDirectory);
    Objects.requireNonNull(memoryTrackingCsvPath);
    Objects.requireNonNull(memoryTrackingSampleInterval);
    if (Files.notExists(tempDirectory)) {
      throw new IllegalArgumentException("does not exist:" + tempDirectory);
    }
    if (!Files.isDirectory(tempDirectory)) {
      throw new IllegalArgumentException("not a directory:" + tempDirectory);
    }
    return new CuVSResourcesImpl(
        tempDirectory, memoryTrackingCsvPath, memoryTrackingSampleInterval);
  }

  @Override
  public BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources) {
    return BruteForceIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources) {
    return CagraIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public FilterBitsetHandle newFilterBitsetHandle(long[] combinedLongs) {
    return new FilterBitsetHandleImpl(combinedLongs);
  }

  @Override
  public MultiPartitionSearchResults searchCagraMultiPartition(
      CuVSResources resources,
      List<CagraIndex> indices,
      CagraQuery query,
      int k,
      FilterBitsetHandle filter)
      throws Throwable {
    return MultiPartitionCagraSearchImpl.search(resources, indices, query, k, filter);
  }

  @Override
  public HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources) {
    return HnswIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public HnswIndex hnswIndexFromCagra(HnswIndexParams hnswParams, CagraIndex cagraIndex)
      throws Throwable {
    return HnswIndexImpl.fromCagra(hnswParams, cagraIndex);
  }

  @Override
  public HnswIndex hnswIndexBuild(
      CuVSResources resources, HnswIndexParams hnswParams, CuVSMatrix dataset) throws Throwable {
    return HnswIndexImpl.build(resources, hnswParams, dataset);
  }

  @Override
  public TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources) {
    return TieredIndexImpl.newBuilder(Objects.requireNonNull(cuVSResources));
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes) {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }
    return CagraIndexImpl.merge(indexes);
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes, CagraIndexParams mergeParams) {
    if (indexes == null || indexes.length == 0) {
      throw new IllegalArgumentException("At least one index must be provided for merging");
    }
    return CagraIndexImpl.merge(indexes, mergeParams);
  }

  @Override
  public GPUInfoProvider gpuInfoProvider() {
    return new GPUInfoProviderImpl();
  }

  @Override
  public CagraIndexParams cagraIndexParamsFromHnswParams(
      long rows,
      long dim,
      int m,
      int efConstruction,
      CagraIndexParams.HnswHeuristicType heuristic,
      CagraIndexParams.CuvsDistanceType metric) {
    try (var nativeCagraIndexParams = createCagraIndexParams();
        var ivfPqIndexParams = createIvfPqIndexParams();
        var ivfPqSearchParams = createIvfPqSearchParams()) {

      // This is already allocated by cuvsCagraIndexParamsCreate,
      // we just need to populate it.
      MemorySegment cuvsIvfPqParamsMemorySegment =
          cuvsCagraIndexParams.graph_build_params(nativeCagraIndexParams.handle());
      cuvsIvfPqParams.ivf_pq_build_params(cuvsIvfPqParamsMemorySegment, ivfPqIndexParams.handle());
      cuvsIvfPqParams.ivf_pq_search_params(
          cuvsIvfPqParamsMemorySegment, ivfPqSearchParams.handle());

      cuvsCagraIndexParams.graph_build_params(
          nativeCagraIndexParams.handle(), cuvsIvfPqParamsMemorySegment);
      checkCuVSError(
          cuvsCagraIndexParamsFromHnswParams(
              nativeCagraIndexParams.handle(),
              rows,
              dim,
              m,
              efConstruction,
              heuristic.value,
              metric.value),
          "cuvsCagraIndexParamsFromHnswParams");

      return populateCagraIndexParamsFromNative(
          nativeCagraIndexParams,
          ivfPqIndexParams,
          ivfPqSearchParams,
          cuvsIvfPqParamsMemorySegment);
    }
  }

  @Override
  public void setLogLevel(Level level) {
    if (level.equals(Level.ALL) || level.equals(Level.FINEST)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_TRACE());
    } else if (level.equals(Level.FINER) || level.equals(Level.FINE)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_DEBUG());
    } else if (level.equals(Level.CONFIG) || level.equals(Level.INFO)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_INFO());
    } else if (level.equals(Level.WARNING)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_WARN());
    } else if (level.equals(Level.SEVERE)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_ERROR());
    } else if (level.equals(Level.OFF)) {
      cuvsSetLogLevel(CUVS_LOG_LEVEL_OFF());
    } else {
      throw new UnsupportedOperationException("Unsupported log level [" + level + "]");
    }
  }

  private static CagraIndexParams populateCagraIndexParamsFromNative(
      CloseableHandle nativeCagraIndexParams,
      CloseableHandle ivfPqIndexParams,
      CloseableHandle ivfPqSearchParams,
      MemorySegment cuvsIvfPqParamsMemorySegment) {
    var algo =
        CagraIndexParams.CagraGraphBuildAlgo.of(
            cuvsCagraIndexParams.build_algo(nativeCagraIndexParams.handle()));
    var builder =
        new CagraIndexParams.Builder()
            .withMetric(
                CagraIndexParams.CuvsDistanceType.of(
                    cuvsCagraIndexParams.metric(nativeCagraIndexParams.handle())))
            .withIntermediateGraphDegree(
                cuvsCagraIndexParams.intermediate_graph_degree(nativeCagraIndexParams.handle()))
            .withGraphDegree(cuvsCagraIndexParams.graph_degree(nativeCagraIndexParams.handle()))
            .withCagraGraphBuildAlgo(algo);

    if (algo == CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT) {
      builder.withNNDescentNumIterations(
          cuvsCagraIndexParams.nn_descent_niter(nativeCagraIndexParams.handle()));
    } else if (algo == CagraIndexParams.CagraGraphBuildAlgo.IVF_PQ) {
      builder.withCuVSIvfPqParams(
          new CuVSIvfPqParams.Builder()
              .withCuVSIvfPqIndexParams(
                  new CuVSIvfPqIndexParams.Builder()
                      .withMaxTrainPointsPerPqCode(
                          cuvsIvfPqIndexParams.max_train_points_per_pq_code(
                              ivfPqIndexParams.handle()))
                      .withAddDataOnBuild(
                          cuvsIvfPqIndexParams.add_data_on_build(ivfPqIndexParams.handle()))
                      .withMetric(
                          CagraIndexParams.CuvsDistanceType.of(
                              cuvsIvfPqIndexParams.metric(ivfPqIndexParams.handle())))
                      .withMetricArg(cuvsIvfPqIndexParams.metric_arg(ivfPqIndexParams.handle()))
                      .withNLists(cuvsIvfPqIndexParams.n_lists(ivfPqIndexParams.handle()))
                      .withKmeansNIters(
                          cuvsIvfPqIndexParams.kmeans_n_iters(ivfPqIndexParams.handle()))
                      .withKmeansTrainsetFraction(
                          cuvsIvfPqIndexParams.kmeans_trainset_fraction(ivfPqIndexParams.handle()))
                      .withPqBits(cuvsIvfPqIndexParams.pq_bits(ivfPqIndexParams.handle()))
                      .withPqDim(cuvsIvfPqIndexParams.pq_dim(ivfPqIndexParams.handle()))
                      .withCodebookKind(
                          CagraIndexParams.CodebookGen.of(
                              cuvsIvfPqIndexParams.codebook_kind(ivfPqIndexParams.handle())))
                      .withForceRandomRotation(
                          cuvsIvfPqIndexParams.force_random_rotation(ivfPqIndexParams.handle()))
                      .withConservativeMemoryAllocation(
                          cuvsIvfPqIndexParams.conservative_memory_allocation(
                              ivfPqIndexParams.handle()))
                      .build())
              .withCuVSIvfPqSearchParams(
                  new CuVSIvfPqSearchParams.Builder()
                      .withNProbes(cuvsIvfPqSearchParams.n_probes(ivfPqSearchParams.handle()))
                      .withLutDtype(
                          CagraIndexParams.CudaDataType.of(
                              cuvsIvfPqSearchParams.lut_dtype(ivfPqSearchParams.handle())))
                      .withInternalDistanceDtype(
                          CagraIndexParams.CudaDataType.of(
                              cuvsIvfPqSearchParams.internal_distance_dtype(
                                  ivfPqSearchParams.handle())))
                      .withPreferredShmemCarveout(
                          cuvsIvfPqSearchParams.preferred_shmem_carveout(
                              ivfPqSearchParams.handle()))
                      .build())
              .withRefinementRate(cuvsIvfPqParams.refinement_rate(cuvsIvfPqParamsMemorySegment))
              .build());
    }
    return builder.build();
  }

  @Override
  public Level getLogLevel() {
    int logLevel = GET_LOG_LEVEL_INVOKER.apply();
    if (logLevel == CUVS_LOG_LEVEL_TRACE()) {
      return Level.ALL;
    } else if (logLevel == CUVS_LOG_LEVEL_DEBUG()) {
      return Level.FINE;
    } else if (logLevel == CUVS_LOG_LEVEL_INFO()) {
      return Level.INFO;
    } else if (logLevel == CUVS_LOG_LEVEL_WARN()) {
      return Level.WARNING;
    } else if (logLevel == CUVS_LOG_LEVEL_ERROR() || logLevel == CUVS_LOG_LEVEL_CRITICAL()) {
      return Level.SEVERE;
    } else if (logLevel == CUVS_LOG_LEVEL_OFF()) {
      return Level.OFF;
    }
    throw new IllegalArgumentException("Unexpected log level [" + logLevel + "]");
  }

  @Override
  public void enableRMMAsyncMemory() {
    checkCuVSError(
        cuvsRMMAsyncMemoryResourceEnableInvoker.apply(), "cuvsRMMAsyncMemoryResourceEnable");
  }

  @Override
  public void enableRMMPooledMemory(int initialPoolSizePercent, int maxPoolSizePercent) {
    checkCuVSError(
        cuvsRMMPoolMemoryResourceEnable(initialPoolSizePercent, maxPoolSizePercent, false),
        "cuvsRMMPoolMemoryResourceEnable");
  }

  @Override
  public void enableRMMManagedPooledMemory(int initialPoolSizePercent, int maxPoolSizePercent) {
    checkCuVSError(
        cuvsRMMPoolMemoryResourceEnable(initialPoolSizePercent, maxPoolSizePercent, true),
        "cuvsRMMPoolMemoryResourceEnable");
  }

  @Override
  public void resetRMMPooledMemory() {
    checkCuVSError(cuvsRMMMemoryResourceResetInvoker.apply(), "cuvsRMMMemoryResourceReset");
  }

  @Override
  public CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long columns, CuVSMatrix.DataType dataType) {

    return new HostMatrixBuilder(size, columns, dataType);
  }

  @Override
  public CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType) {

    return new HostMatrixBuilder(size, columns, rowStride, columnStride, dataType);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources resources, long size, long columns, CuVSMatrix.DataType dataType) {
    var rowBytes = columns * dataType.bytes();
    return rowBytes > PinnedMemoryBuffer.CHUNK_BYTES
        ? new DirectDeviceMatrixBuilder(resources, size, columns, dataType)
        : new BufferedDeviceMatrixBuilder(resources, size, columns, dataType);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources resources,
      long size,
      long columns,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType) {
    var rowBytes = columns * dataType.bytes();
    return rowBytes > PinnedMemoryBuffer.CHUNK_BYTES
        ? new DirectDeviceMatrixBuilder(resources, size, columns, rowStride, columnStride, dataType)
        : new BufferedDeviceMatrixBuilder(
            resources, size, columns, rowStride, columnStride, dataType);
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    return createNativeDataset$mh;
  }

  @Override
  public MethodHandle newNativeMatrixBuilderWithStrides() {
    return createNativeDatasetWithStrides$mh;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(float[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.FLOAT);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(int[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.INT);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  @Override
  public CuVSMatrix newMatrixFromArray(byte[][] vectors) {
    Objects.requireNonNull(vectors);
    if (vectors.length == 0) {
      throw new IllegalArgumentException("vectors should not be empty");
    }
    int size = vectors.length;
    int columns = vectors[0].length;

    var dataset = new CuVSHostMatrixArenaImpl(size, columns, CuVSMatrix.DataType.BYTE);
    Util.copy(dataset.memorySegment(), vectors);
    return dataset;
  }

  private abstract static class MatrixBuilder<T extends CuVSMatrixInternal> {

    protected final long columns;
    protected final long size;
    protected final T matrix;
    protected final long elementSize;
    protected final long rowSize;
    protected final long rowBytes;
    protected int currentRow;

    protected MatrixBuilder(T matrix, long size, long columns) {
      this.columns = columns;
      this.size = size;
      this.matrix = matrix;
      this.elementSize = matrix.valueLayout().byteSize();
      this.rowSize = columns * elementSize;
      this.rowBytes = rowSize;
      this.currentRow = 0;
    }

    protected MatrixBuilder(T matrix, long size, long columns, int rowStride) {
      this.columns = columns;
      this.size = size;
      this.matrix = matrix;
      this.elementSize = matrix.valueLayout().byteSize();
      this.rowSize = rowStride > 0 ? rowStride * elementSize : columns * elementSize;
      this.rowBytes = columns * elementSize;

      this.currentRow = 0;
    }

    public void addVector(float[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    public void addVector(byte[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    public void addVector(int[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    public void addVector(short[] vector) {
      if (vector.length != columns) {
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT, "Expected a vector of size [%d], got [%d]", columns, vector.length));
      }
      internalAddVector(MemorySegment.ofArray(vector));
    }

    protected abstract void internalAddVector(MemorySegment vector);
  }

  /**
   * This {@link CuVSDeviceMatrix} builder implementation returns a {@link CuVSDeviceMatrix} backed by managed RMM
   * device memory. It uses a {@link PinnedMemoryBuffer} to batch data before copying it to the GPU.
   */
  private static final class BufferedDeviceMatrixBuilder extends MatrixBuilder<CuVSDeviceMatrixImpl>
      implements CuVSMatrix.Builder<CuVSDeviceMatrix> {

    private final MemorySegment stream;

    private final CuVSResources resources;
    private final long bufferRowCount;
    private int currentBufferRow;

    private BufferedDeviceMatrixBuilder(
        CuVSResources resources, long size, long columns, CuVSMatrix.DataType dataType) {
      super(CuVSDeviceMatrixRMMImpl.create(resources, size, columns, dataType), size, columns);
      this.stream = Util.getStream(resources);
      this.resources = resources;

      this.bufferRowCount = Math.min((PinnedMemoryBuffer.CHUNK_BYTES / rowBytes), size);
      this.currentBufferRow = 0;
    }

    private BufferedDeviceMatrixBuilder(
        CuVSResources resources,
        long size,
        long columns,
        int rowStride,
        int columnStride,
        CuVSMatrix.DataType dataType) {
      super(
          CuVSDeviceMatrixRMMImpl.create(
              resources, size, columns, rowStride, columnStride, dataType),
          size,
          columns,
          rowStride);

      this.stream = Util.getStream(resources);
      this.resources = resources;

      this.bufferRowCount = Math.min((PinnedMemoryBuffer.CHUNK_BYTES / rowBytes), size);
      this.currentBufferRow = 0;
    }

    @Override
    protected void internalAddVector(MemorySegment vector) {
      if (currentRow >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }
      var hostBufferOffset = currentBufferRow * rowBytes;
      try (var access = resources.access()) {
        var hostBuffer = CuVSResourcesImpl.getHostBuffer(access);
        MemorySegment.copy(vector, 0, hostBuffer, hostBufferOffset, rowBytes);

        currentRow++;
        currentBufferRow++;
        if (currentBufferRow == bufferRowCount) {
          flushBuffer(hostBuffer);
        }
      }
    }

    private void flushBuffer(MemorySegment hostBuffer) {
      if (currentBufferRow > 0) {
        var deviceMemoryOffset = (currentRow - currentBufferRow) * rowSize;
        var dst = matrix.memorySegment().asSlice(deviceMemoryOffset);
        checkCudaError(
            cudaMemcpy2DAsync(
                dst,
                rowSize,
                hostBuffer,
                rowBytes,
                rowBytes,
                currentBufferRow,
                CudaMemcpyKind.HOST_TO_DEVICE.kind,
                stream),
            "cudaMemcpy2DAsync");

        currentBufferRow = 0;
        checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      }
    }

    @Override
    public CuVSDeviceMatrix build() {
      try (var access = resources.access()) {
        var hostBuffer = CuVSResourcesImpl.getHostBuffer(access);
        flushBuffer(hostBuffer);
      }
      return matrix;
    }
  }

  /**
   * This {@link CuVSDeviceMatrix} builder implementation returns a {@link CuVSDeviceMatrix} backed by managed RMM
   * device memory. It uses a {@link PinnedMemoryBuffer} to batch data before copying it to the GPU.
   */
  private static final class DirectDeviceMatrixBuilder extends MatrixBuilder<CuVSDeviceMatrixImpl>
      implements CuVSMatrix.Builder<CuVSDeviceMatrix> {

    private final MemorySegment stream;

    private int currentRow;

    private DirectDeviceMatrixBuilder(
        CuVSResources resources, long size, long columns, CuVSMatrix.DataType dataType) {
      super(CuVSDeviceMatrixRMMImpl.create(resources, size, columns, dataType), size, columns);
      this.stream = Util.getStream(resources);
      this.currentRow = 0;
    }

    private DirectDeviceMatrixBuilder(
        CuVSResources resources,
        long size,
        long columns,
        int rowStride,
        int columnStride,
        CuVSMatrix.DataType dataType) {
      super(
          CuVSDeviceMatrixRMMImpl.create(
              resources, size, columns, rowStride, columnStride, dataType),
          size,
          columns,
          rowStride);

      this.stream = Util.getStream(resources);
      this.currentRow = 0;
    }

    @Override
    protected void internalAddVector(MemorySegment vector) {
      if (currentRow >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }

      var deviceMemoryOffset = currentRow * rowSize;
      var dst = matrix.memorySegment().asSlice(deviceMemoryOffset);
      Util.cudaMemcpyAsync(dst, vector, rowBytes, CudaMemcpyKind.HOST_TO_DEVICE, stream);
      checkCudaError(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      currentRow++;
    }

    @Override
    public CuVSDeviceMatrix build() {
      return matrix;
    }
  }

  private static class HostMatrixBuilder extends MatrixBuilder<CuVSHostMatrixImpl>
      implements CuVSMatrix.Builder<CuVSHostMatrix> {

    private HostMatrixBuilder(long size, long columns, CuVSMatrix.DataType dataType) {
      super(new CuVSHostMatrixArenaImpl(size, columns, dataType), size, columns);
    }

    private HostMatrixBuilder(
        long size, long columns, int rowStride, int columnStride, CuVSMatrix.DataType dataType) {
      super(
          new CuVSHostMatrixArenaImpl(size, columns, rowStride, columnStride, dataType),
          size,
          columns,
          rowStride);
    }

    @Override
    protected void internalAddVector(MemorySegment vector) {
      if (currentRow >= size) {
        throw new ArrayIndexOutOfBoundsException();
      }
      MemorySegment.copy(vector, 0, matrix.memorySegment(), ((currentRow++) * rowSize), rowBytes);
    }

    @Override
    public CuVSHostMatrix build() {
      return matrix;
    }
  }
}
