/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CodebookGen;
import com.nvidia.cuvs.CagraIndexParams.CudaDataType;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CuVSIvfPqIndexParams;
import com.nvidia.cuvs.CuVSIvfPqParams;
import com.nvidia.cuvs.CuVSIvfPqSearchParams;

/**
 * A centralized place for producing {@link CagraIndexParams} from the cuvs-lucene input parameter
 * classes based on the chosen strategy.
 *
 * <p>For the {@code HEURISTIC} strategy the build heuristics are delegated to cuVS.
 *
 */
public class CagraIndexParamsFactory {

  private CagraIndexParamsFactory() {}

  /**
   * Translation of the internal logic found in the constructor of {@code struct ivf_pq_params} in
   * cuVS's {@code cpp/include/cuvs/neighbors/ivf_pq.hpp}.
   *
   * Ideally we should hook into the internal API but this is currently replicated to avoid complications
   * in other parts of code base.
   *
   * TODO: cuvs-java has no standalone binding for that C++ constructor -- only fromHnswParams and
   * fromDataset, which bundle algorithm selection together with parameter tuning and offer no way
   * to pin the algorithm while still getting cuVS's own auto-tuned IVF-PQ parameters. If cuvs-java
   * exposed the ivf_pq_params(dataset_extents, metric) constructor directly (or an equivalent
   * narrower entry point that derives IVF-PQ parameters without also choosing the algorithm), this
   * method could delegate to it instead of reimplementing the heuristic here.
   */
  private static CuVSIvfPqParams getCuVSIvfPqParams(long rows, long dimension) {
    int pqDim;
    int pqBits;
    if (dimension <= 32) {
      pqDim = 16;
      pqBits = 8;
    } else {
      pqBits = 4;
      if (dimension <= 64) {
        pqDim = 32;
      } else if (dimension <= 128) {
        pqDim = 64;
      } else if (dimension <= 192) {
        pqDim = 96;
      } else {
        pqDim = (int) roundUpSafe(dimension / 2, 128);
      }
    }
    int nLists = (int) Math.max(1, rows / 2000);
    final int kmeansNIters = 10;
    final double kMinPointsPerCluster = 32;
    double minKmeansTrainsetPoints = kMinPointsPerCluster * nLists;
    final double maxKmeansTrainsetFraction = 1.0;
    double minKmeansTrainsetFraction =
        Math.min(maxKmeansTrainsetFraction, minKmeansTrainsetPoints / rows);
    double kmeansTrainsetFraction =
        Math.clamp(
            1.0 / Math.sqrt(rows * 1e-5), minKmeansTrainsetFraction, maxKmeansTrainsetFraction);
    int nProbes = (int) Math.round(Math.sqrt(nLists) / 20 + 4);
    CuVSIvfPqIndexParams cuVSIvfPqIndexParams =
        new CuVSIvfPqIndexParams.Builder()
            .withCodebookKind(CodebookGen.PER_SUBSPACE)
            .withKmeansNIters(kmeansNIters)
            .withKmeansTrainsetFraction(kmeansTrainsetFraction)
            .withNLists(nLists)
            .withPqBits(pqBits)
            .withPqDim(pqDim)
            .withAddDataOnBuild(true)
            .withConservativeMemoryAllocation(true)
            .build();
    CuVSIvfPqSearchParams cuVSIvfPqSearchParams =
        new CuVSIvfPqSearchParams.Builder()
            .withLutDtype(CudaDataType.CUDA_R_16F)
            .withInternalDistanceDtype(CudaDataType.CUDA_R_16F)
            .withNProbes(nProbes)
            .build();
    return new CuVSIvfPqParams.Builder()
        .withCuVSIvfPqIndexParams(cuVSIvfPqIndexParams)
        .withCuVSIvfPqSearchParams(cuVSIvfPqSearchParams)
        .withRefinementRate(1)
        .build();
  }

  private static long roundUpSafe(long numberToRound, long modulus) {
    long remainder = numberToRound % modulus;
    if (remainder == 0) {
      return numberToRound;
    }
    return numberToRound - remainder + modulus;
  }

  private static CagraIndexParams getNNDescentParams(
      int graphDegree,
      int intGraphDegree,
      int writerThreads,
      long nnDescentNumIterations,
      CuvsDistanceType cuvsDistanceType) {
    return new CagraIndexParams.Builder()
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
        .withGraphDegree(graphDegree)
        .withIntermediateGraphDegree(intGraphDegree)
        .withNNDescentNumIterations(nnDescentNumIterations)
        .withNumWriterThreads(writerThreads)
        .withMetric(cuvsDistanceType)
        .build();
  }

  /**
   * @param explicitIvfPqParams the caller's own IVF-PQ params if they set one ({@link
   *     AcceleratedHNSWParams#isCuVSIvfPqParamsExplicit()}), honored as-is; otherwise {@code
   *     null}, in which case params are auto-tuned from {@code rows}/{@code dimension}.
   */
  private static CagraIndexParams getIVFPQParams(
      int graphDegree,
      int intGraphDegree,
      int writerThreads,
      long rows,
      long dimension,
      CuvsDistanceType cuvsDistanceType,
      CuVSIvfPqParams explicitIvfPqParams) {
    return new CagraIndexParams.Builder()
        .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.IVF_PQ)
        .withCuVSIvfPqParams(
            explicitIvfPqParams != null ? explicitIvfPqParams : getCuVSIvfPqParams(rows, dimension))
        .withNumWriterThreads(writerThreads)
        .withIntermediateGraphDegree(intGraphDegree)
        .withGraphDegree(graphDegree)
        .withMetric(cuvsDistanceType)
        .build();
  }

  /**
   * Creates an instance of {@link CagraIndexParams} for the GPU-native CAGRA index based on the
   * chosen strategy in the {@link GPUSearchParams}.
   *
   * @param gpuSearchParams the input parameters for the build and search on the GPU API
   * @param rows number of vectors in the data set
   * @param dimension the dimension of the vectors in the data set
   * @return an instance of {@link CagraIndexParams}
   */
  public static CagraIndexParams create(
      GPUSearchParams gpuSearchParams, long rows, long dimension) {
    if (gpuSearchParams.getStrategy().equals(GPUSearchParams.Strategy.HEURISTIC)) {
      // Delegate the build-algorithm choice and its parameters to cuVS' dataset heuristic, which
      // switches on the row count and tunes the algorithm with the caller's build quality.
      CagraIndexParams derived =
          CagraIndexParams.fromDataset(
              rows,
              dimension,
              gpuSearchParams.getGraphdegree(),
              gpuSearchParams.getCuvsDistanceType(),
              gpuSearchParams.getBuildQuality());
      return new CagraIndexParams.Builder()
          .withGraphDegree(derived.getGraphDegree())
          .withIntermediateGraphDegree(derived.getIntermediateGraphDegree())
          .withCagraGraphBuildAlgo(derived.getCagraGraphBuildAlgo())
          .withCuVSIvfPqParams(derived.getCuVSIvfPqParams())
          .withNNDescentNumIterations(derived.getNNDescentNumIterations())
          .withMetric(gpuSearchParams.getCuvsDistanceType())
          .withNumWriterThreads(gpuSearchParams.getWriterThreads())
          .build();
    }
    // CUSTOM: forward the caller's algorithm and the parameters it consumes -- IVF-PQ params for
    // IVF_PQ, nn-descent iterations for NN_DESCENT (each is ignored by the other algorithm).
    return new CagraIndexParams.Builder()
        .withGraphDegree(gpuSearchParams.getGraphdegree())
        .withIntermediateGraphDegree(gpuSearchParams.getIntermediateGraphDegree())
        .withCagraGraphBuildAlgo(gpuSearchParams.getCagraGraphBuildAlgo())
        .withCuVSIvfPqParams(gpuSearchParams.getCuVSIvfPqParams())
        .withNNDescentNumIterations(gpuSearchParams.getnNDescentNumIterations())
        .withMetric(gpuSearchParams.getCuvsDistanceType())
        .withNumWriterThreads(gpuSearchParams.getWriterThreads())
        .build();
  }

  /**
   * Creates an instance of {@link CagraIndexParams} for the accelerated-HNSW index based on the
   * chosen strategy in the {@link AcceleratedHNSWParams}.
   *
   * @param acceleratedHNSWParams the input parameters for the build on the GPU API
   * @param rows number of vectors in the data set
   * @param dimension the dimension of the vectors in the data set
   * @return an instance of {@link CagraIndexParams}
   */
  public static CagraIndexParams create(
      AcceleratedHNSWParams acceleratedHNSWParams, long rows, long dimension) {
    if (acceleratedHNSWParams.getStrategy().equals(AcceleratedHNSWParams.Strategy.HEURISTIC)) {
      // An explicit cagraGraphBuildAlgo of IVF_PQ or NN_DESCENT overrides the heuristic choice;
      // AUTO_SELECT (the default) delegates all parameter derivation to cuVS via fromHnswParams.
      CagraGraphBuildAlgo algo = acceleratedHNSWParams.getCagraGraphBuildAlgo();
      if (algo == CagraGraphBuildAlgo.IVF_PQ) {
        return getIVFPQParams(
            acceleratedHNSWParams.getGraphdegree(),
            acceleratedHNSWParams.getIntermediateGraphDegree(),
            acceleratedHNSWParams.getWriterThreads(),
            rows,
            dimension,
            acceleratedHNSWParams.getCuvsDistanceType(),
            acceleratedHNSWParams.isCuVSIvfPqParamsExplicit()
                ? acceleratedHNSWParams.getCuVSIvfPqParams()
                : null);
      } else if (algo == CagraGraphBuildAlgo.NN_DESCENT) {
        return getNNDescentParams(
            acceleratedHNSWParams.getGraphdegree(),
            acceleratedHNSWParams.getIntermediateGraphDegree(),
            acceleratedHNSWParams.getWriterThreads(),
            acceleratedHNSWParams.getNNDescentNumIterations(),
            acceleratedHNSWParams.getCuvsDistanceType());
      } else {
        // AUTO_SELECT: delegate the derivation of the graph degrees, build algorithm and its
        // parameters to cuVS, expressed in terms of the HNSW-equivalent maxConn/beamWidth.
        CagraIndexParams derived =
            CagraIndexParams.fromHnswParams(
                rows,
                dimension,
                acceleratedHNSWParams.getMaxConn(),
                acceleratedHNSWParams.getBeamWidth(),
                acceleratedHNSWParams.getHnswHeuristicType(),
                acceleratedHNSWParams.getCuvsDistanceType());
        // TODO: fromHnswParams has no writerThreads argument, so its result carries the cuVS
        // default (not a heuristic value). We can rebuild the CagraIndexParams with the
        // caller-supplied writerThreads for now but should fix this in cuVS in the future.
        return new CagraIndexParams.Builder()
            .withGraphDegree(derived.getGraphDegree())
            .withIntermediateGraphDegree(derived.getIntermediateGraphDegree())
            .withCagraGraphBuildAlgo(derived.getCagraGraphBuildAlgo())
            .withCuVSIvfPqParams(derived.getCuVSIvfPqParams())
            .withNNDescentNumIterations(derived.getNNDescentNumIterations())
            .withMetric(acceleratedHNSWParams.getCuvsDistanceType())
            .withNumWriterThreads(acceleratedHNSWParams.getWriterThreads())
            .build();
      }
    }
    // CUSTOM: forward the caller's algorithm and the parameters it consumes.
    return new CagraIndexParams.Builder()
        .withGraphDegree(acceleratedHNSWParams.getGraphdegree())
        .withIntermediateGraphDegree(acceleratedHNSWParams.getIntermediateGraphDegree())
        .withCagraGraphBuildAlgo(acceleratedHNSWParams.getCagraGraphBuildAlgo())
        .withCuVSIvfPqParams(acceleratedHNSWParams.getCuVSIvfPqParams())
        .withNNDescentNumIterations(acceleratedHNSWParams.getNNDescentNumIterations())
        .withMetric(acceleratedHNSWParams.getCuvsDistanceType())
        .withNumWriterThreads(acceleratedHNSWParams.getWriterThreads())
        .build();
  }
}
