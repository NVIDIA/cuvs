/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraSearchParams.SearchAlgo;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Diagnostic (not a regression test): measures how often a filtered multi-partition GPU search
 * returns fewer than k hits (and therefore falls back to the per-segment path) for the same index
 * shape as {@link TestMultiSegmentGPUFilterConcurrency}, across several CAGRA itopk/algo settings.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestDiagFilteredMultiPartitionShortfall extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vectors";
  private static final String CATEGORY_FIELD = "cat";

  @BeforeClass
  public static void beforeClass() {
    try {
      CuVSProvider.provider().enableRMMAsyncMemory();
    } catch (UnsupportedOperationException unsupported) {
      assumeTrue("cuVS not supported: " + unsupported.getMessage(), false);
    }
    assumeTrue("cuVS not supported", isSupported());
  }

  private record Config(int iTopK, SearchAlgo algo) {}

  @Test
  public void measureShortfallRate() throws Exception {
    int trials = Integer.getInteger("diag.trials", 6);
    int numQueries = Integer.getInteger("diag.queries", 64);
    int[] numCategoriesOptions = {24};
    final int datasetSize = 2000;
    final int dimensions = 128;
    final int topK = 10;

    Config[] configs = {
      new Config(10, SearchAlgo.AUTO),
      new Config(10, SearchAlgo.MULTI_CTA),
      new Config(10, SearchAlgo.SINGLE_CTA),
      new Config(32, SearchAlgo.AUTO),
      new Config(64, SearchAlgo.AUTO),
      new Config(128, SearchAlgo.AUTO),
      new Config(256, SearchAlgo.AUTO),
    };

    Codec codec = new CuVS2510GPUSearchCodec();
    Random rnd = random();
    for (int numCategories : numCategoriesOptions) {
      Map<Config, long[]> totals = new LinkedHashMap<>();
      for (Config c : configs) totals.put(c, new long[2]);
      for (int trial = 0; trial < trials; trial++) {
        try (Directory directory = new ByteBuffersDirectory()) {
          IndexWriterConfig config =
              new IndexWriterConfig().setCodec(codec).setMergePolicy(NoMergePolicy.INSTANCE);
          float[][] dataset = generateDataset(rnd, datasetSize, dimensions);
          try (IndexWriter writer = new IndexWriter(directory, config)) {
            final int commitEvery = datasetSize / 4;
            for (int i = 0; i < datasetSize; i++) {
              Document doc = new Document();
              doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
              doc.add(new StringField(CATEGORY_FIELD, "c" + (i % numCategories), Field.Store.NO));
              doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
              writer.addDocument(doc);
              if ((i + 1) % commitEvery == 0) {
                writer.commit();
              }
            }
            writer.commit();
          }
          try (DirectoryReader reader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            float[][] queries = generateDataset(rnd, numQueries, dimensions);
            for (Config cfg : configs) {
              long[] t = totals.get(cfg);
              long trialFallbacks = 0;
              for (int c = 0; c < numCategories; c++) {
                Query filter = new TermQuery(new Term(CATEGORY_FIELD, "c" + c));
                for (float[] q : queries) {
                  GPUKnnFloatVectorQuery query =
                      new GPUKnnFloatVectorQuery(
                          VECTOR_FIELD, q, topK, filter, cfg.iTopK(), 1, 0, 0, cfg.algo());
                  boolean gpu = searcher.rewrite(query).toString().contains("GPUDocAndScoreQuery");
                  t[0]++;
                  if (!gpu) {
                    t[1]++;
                    trialFallbacks++;
                  }
                }
              }
              System.out.println(
                  "DIAG trial="
                      + trial
                      + " categories="
                      + numCategories
                      + " itopk="
                      + cfg.iTopK()
                      + " algo="
                      + cfg.algo()
                      + " shortfalls="
                      + trialFallbacks
                      + "/"
                      + (numCategories * queries.length)
                      + " anyShortfallPer24Filters="
                      + (trialFallbacks > 0));
            }
          }
        }
      }
      for (Map.Entry<Config, long[]> e : totals.entrySet()) {
        long[] t = e.getValue();
        System.out.printf(
            "DIAG-SUMMARY categories=%d itopk=%d algo=%s shortfalls=%d/%d (%.4f%%)%n",
            numCategories, e.getKey().iTopK(), e.getKey().algo(), t[1], t[0], 100.0 * t[1] / t[0]);
      }
    }
  }
}
