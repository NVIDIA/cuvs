/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Random;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** CAGRA builds from host datasets whose rows are padded or carry a per-row trailer. */
@RunWith(RandomizedRunner.class)
public class StridedDatasetIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
  }

  /**
   * A host dataset whose row stride equals its column count, and one whose rows carry a 4-byte
   * trailer (Lucene's int8 layout), both build without a packed copy. Odd rows get a large trailer
   * and even rows a zero one: were the trailer read as part of the vectors, graph neighbours would
   * split by row parity instead of being half odd and half even.
   */
  @Test
  public void testCagraBuildFromStridedHostDataset() throws Throwable {
    int rows = 20_000, dims = 64, pitch = dims + 4;
    var rnd = new Random(random.nextLong());
    try (var arena = Arena.ofConfined();
        var resources = CheckedCuVSResources.create()) {
      var strided = arena.allocate((long) rows * pitch, 64);
      var packed = arena.allocate((long) rows * dims, 64);
      byte[][] centres = new byte[200][dims];
      for (var centre : centres) rnd.nextBytes(centre);
      int[] cluster = new int[rows];
      for (int r = 0; r < rows; ++r) {
        cluster[r] = rnd.nextInt(centres.length);
        var vector = new byte[dims];
        for (int c = 0; c < dims; ++c) {
          vector[c] =
              (byte) Math.max(-128, Math.min(127, centres[cluster[r]][c] + rnd.nextInt(17) - 8));
        }
        MemorySegment.copy(vector, 0, strided, ValueLayout.JAVA_BYTE, (long) r * pitch, dims);
        MemorySegment.copy(vector, 0, packed, ValueLayout.JAVA_BYTE, (long) r * dims, dims);
        strided.set(ValueLayout.JAVA_INT_UNALIGNED, (long) r * pitch + dims, 0x7f7f7f7f * (r & 1));
      }
      MethodHandle strides = CuVSProvider.provider().newNativeMatrixBuilderWithStrides();
      MethodHandle dense = CuVSProvider.provider().newNativeMatrixBuilder();
      var stridedMatrix =
          (CuVSMatrix)
              strides.invokeExact(strided, rows, dims, pitch, -1, CuVSMatrix.DataType.BYTE);
      var denseWithStride =
          (CuVSMatrix) strides.invokeExact(packed, rows, dims, dims, -1, CuVSMatrix.DataType.BYTE);
      var denseMatrix =
          (CuVSMatrix) dense.invokeExact(packed, rows, dims, CuVSMatrix.DataType.BYTE);

      for (var matrix : new CuVSMatrix[] {denseMatrix, denseWithStride, stridedMatrix}) {
        double[] stats = neighbourStats(resources, matrix, cluster);
        assertTrue("same-cluster fraction " + stats[0], stats[0] > 0.95);
        assertTrue("same-parity fraction " + stats[1], stats[1] > 0.4 && stats[1] < 0.6);
      }
    }
  }

  /** Over all graph edges: {fraction joining nodes of the same cluster, of the same row parity}. */
  private static double[] neighbourStats(CuVSResources resources, CuVSMatrix dataset, int[] cluster)
      throws Throwable {
    var params =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(16)
            .withIntermediateGraphDegree(32)
            .withMetric(CagraIndexParams.CuvsDistanceType.L2Expanded)
            .build();
    try (var index =
            CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build();
        var graph = index.getGraph();
        var hostGraph = graph.toHost()) {
      long edges = 0, sameCluster = 0, sameParity = 0;
      var neighbours = new int[(int) hostGraph.columns()];
      for (int node = 0; node < hostGraph.size(); ++node) {
        hostGraph.getRow(node).toArray(neighbours);
        for (int n : neighbours) {
          edges++;
          if (cluster[n] == cluster[node]) sameCluster++;
          if ((n & 1) == (node & 1)) sameParity++;
        }
      }
      return new double[] {sameCluster / (double) edges, sameParity / (double) edges};
    }
  }
}
