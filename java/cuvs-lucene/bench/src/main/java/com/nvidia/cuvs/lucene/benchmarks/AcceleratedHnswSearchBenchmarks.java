/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene.benchmarks;

import static com.nvidia.cuvs.lucene.benchmarks.Utils.cleanup;
import static com.nvidia.cuvs.lucene.benchmarks.Utils.generateDataset;
import static com.nvidia.cuvs.lucene.benchmarks.Utils.index;
import static com.nvidia.cuvs.lucene.benchmarks.Utils.search;

import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@BenchmarkMode(Mode.All)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1)
@Warmup(iterations = 0, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
public class AcceleratedHnswSearchBenchmarks {

  @SuppressWarnings("unused")
  private static Logger log = Logger.getLogger(AcceleratedHnswSearchBenchmarks.class.getName());

  private static Random random;
  private static Path indexDirPath;
  private static Codec codec;

  private final int COMMIT_FREQ = 100;
  private final String ID_FIELD = "id";
  private final String VECTOR_FIELD = "vector_field";

  private int numDocs;
  private int dimension;
  private int topK;
  private float[][] dataset;
  private float[] queryVector;

  @Setup(Level.Trial)
  public void setup() throws Exception {
    random = new Random(222);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
    codec = new Lucene101AcceleratedHNSWCodec(32, 128, 64, 3, 16, 100);
    numDocs = 1000;
    dimension = 128;
    topK = 5;
    dataset = generateDataset(random, numDocs, dimension);
    queryVector = generateDataset(random, 1, dimension)[0];
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);
    index(indexDirPath, config, dataset, numDocs, VECTOR_FIELD, ID_FIELD, COMMIT_FREQ);
  }

  @Benchmark
  public void benchmarkAcceleratedHnswSearch(Blackhole blackhole) throws Exception {
    KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD, queryVector, topK);
    search(indexDirPath, VECTOR_FIELD, queryVector, topK, query);
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    cleanup(indexDirPath);
  }
}
