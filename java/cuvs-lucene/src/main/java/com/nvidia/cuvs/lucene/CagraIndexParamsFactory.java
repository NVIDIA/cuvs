/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams;

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
      // Delegate the derivation of the graph degrees, build algorithm and its parameters to cuVS,
      // expressed in terms of the HNSW-equivalent maxConn/beamWidth.
      CagraIndexParams derived =
          CagraIndexParams.fromHnswParams(
              rows,
              dimension,
              acceleratedHNSWParams.getMaxConn(),
              acceleratedHNSWParams.getBeamWidth(),
              acceleratedHNSWParams.getHnswHeuristicType(),
              acceleratedHNSWParams.getCuvsDistanceType());
      // TODO: fromHnswParams has no writerThreads argument, so its result carries the cuVS default
      // (not a heuristic value). We can rebuild the CagraIndexParams with the caller-supplied
      // writerThreads for now but should fix this in cuVS in the future.
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
