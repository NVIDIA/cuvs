/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.foreign.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Fork(value = 1, warmups = 0)
@State(Scope.Benchmark)
public class CagraIndexBenchmarks {

    @Param({ "1024" })
    private int dims;

    @Param({ "100" })
    private int size;

    private float[][] arrayDataset;

    private Arena arena;

    private MemorySegment memorySegmentDataset;

    private static final Random random = new Random();

    private static float[][] createSampleData(int size, int dimensions) {
        var array = new float[size][dimensions];
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < dimensions; ++j) {
                array[i][j] = random.nextFloat();
            }
        }
        return array;
    }

    private static MemorySegment createSampleDataSegment(Arena arena, float[][] array, int size, int dimensions) {
        final ValueLayout.OfFloat C_FLOAT = (ValueLayout.OfFloat) Linker.nativeLinker().canonicalLayouts().get("float");

        MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout((long)size * dimensions, C_FLOAT);

        var segment = arena.allocate(dataMemoryLayout);
        for (int i = 0; i < size; ++i) {
            var vector = array[i];
            MemorySegment.copy(vector, 0, segment, C_FLOAT, (i * dimensions * C_FLOAT.byteSize()), dimensions);
        }
        return segment;
    }

    private static CuVSMatrix fromMemorySegment(MemorySegment memorySegment, int size, int dimensions) {
        try {
            return (CuVSMatrix)
                    CuVSProvider.provider()
                            .newNativeMatrixBuilder()
                            .invokeExact(memorySegment, size, dimensions, CuVSMatrix.DataType.FLOAT);
        } catch (Throwable e) {
            if (e instanceof Error err) {
                throw err;
            } else if (e instanceof RuntimeException re) {
                throw re;
            } else {
                throw new RuntimeException(e);
            }
        }
    }

    @Setup
    public void initialize() {
        arena = Arena.ofShared();
        arrayDataset = createSampleData(size, dims);
        memorySegmentDataset = createSampleDataSegment(arena, arrayDataset, size, dims);
    }

    @TearDown
    public void cleanUp() {
        if (arena != null) {
            arena.close();
        }
    }

    @Benchmark
    public void testIndexingAndSerializeToFile() throws Throwable {
        try (CuVSResources resources = CuVSResources.create()) {
            // Configure index parameters
            CagraIndexParams indexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(1)
                .withIntermediateGraphDegree(2)
                .withNumWriterThreads(32)
                .withMetric(CagraIndexParams.CuvsDistanceType.L2Expanded)
                .build();

            // Create the index with the dataset
            try (CagraIndex index = CagraIndex.newBuilder(resources)
                .fromDataset(arrayDataset)
                .withIndexParams(indexParams)
                .build()) {

              // Saving the index on to the disk.
              var indexFilePath = Path.of(UUID.randomUUID() + ".cag");
              try (var outputStream = Files.newOutputStream(indexFilePath)) {
                index.serialize(outputStream);
              }

              // Cleanup
              Files.deleteIfExists(indexFilePath);
            }
        }
    }

    @Benchmark
    public void testIndexingFromHeap(Blackhole blackhole) throws Throwable {
        try (CuVSResources resources = CuVSResources.create()) {

            // Configure index parameters
            CagraIndexParams indexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(1)
                .withIntermediateGraphDegree(2)
                .withNumWriterThreads(32)
                .withMetric(CagraIndexParams.CuvsDistanceType.L2Expanded)
                .build();

            // Create the index with the dataset
            CagraIndex index = CagraIndex.newBuilder(resources)
                .fromDataset(arrayDataset)
                .withIndexParams(indexParams)
                .build();
            blackhole.consume(index);
        }
    }

    @Benchmark
    public void testIndexingFromMemorySegment(Blackhole blackhole) throws Throwable {
        try (CuVSResources resources = CuVSResources.create()) {
            // Configure index parameters
            CagraIndexParams indexParams = new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(1)
                .withIntermediateGraphDegree(2)
                .withNumWriterThreads(32)
                .withMetric(CagraIndexParams.CuvsDistanceType.L2Expanded)
                .build();

            // Create the index with the dataset
            CagraIndex index = CagraIndex.newBuilder(resources)
                .fromDataset(fromMemorySegment(memorySegmentDataset, size, dims))
                .withIndexParams(indexParams)
                .build();
            blackhole.consume(index);
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @BenchmarkMode(Mode.AverageTime)
    public void testDatasetFromHeap(Blackhole blackhole) throws Throwable {
        try (var dataset = CuVSMatrix.ofArray(arrayDataset)) {
            blackhole.consume(dataset);
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @BenchmarkMode(Mode.AverageTime)
    public void testDatasetFromMemorySegment(Blackhole blackhole) throws Throwable {
        try (var dataset = fromMemorySegment(memorySegmentDataset, size, dims)) {
            blackhole.consume(dataset);
        }
    }
}
