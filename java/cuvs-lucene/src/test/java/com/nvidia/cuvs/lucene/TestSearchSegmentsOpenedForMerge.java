/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.tests.util.TestUtil.alwaysKnnVectorsFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.ConcurrentMergeScheduler;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * IndexWriter pools the {@link org.apache.lucene.index.SegmentReader} it opens to merge a segment
 * and hands that same reader (or a clone sharing its core) to near-real-time readers opened while
 * the merge is still running. Searching such a reader must work even though its vectors reader was
 * opened with a merge {@link org.apache.lucene.store.IOContext}.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestSearchSegmentsOpenedForMerge extends LuceneTestCase {

  private static final int NUM_SEGMENTS = 3;
  private static final int DOCS_PER_SEGMENT = 64;
  private static final int DIMENSION = 16;

  @BeforeClass
  public static void beforeClass() {
    assumeTrue("cuVS not supported", isSupported());
  }

  /** Forced merge of all segments that blocks once its input readers are open. */
  private static final class BlockingMergePolicy extends FilterMergePolicy {
    final CountDownLatch mergeReadersOpened = new CountDownLatch(1);
    final CountDownLatch releaseMerge = new CountDownLatch(1);

    BlockingMergePolicy() {
      super(NoMergePolicy.INSTANCE);
    }

    @Override
    public MergeSpecification findForcedMerges(
        SegmentInfos segmentInfos,
        int maxSegmentCount,
        Map<SegmentCommitInfo, Boolean> segmentsToMerge,
        MergeContext mergeContext) {
      List<SegmentCommitInfo> segments = new ArrayList<>();
      for (SegmentCommitInfo info : segmentInfos) {
        if (segmentsToMerge.containsKey(info)
            && !mergeContext.getMergingSegments().contains(info)) {
          segments.add(info);
        }
      }
      if (segments.size() < 2) {
        return null;
      }
      MergeSpecification spec = new MergeSpecification();
      spec.add(
          new OneMerge(segments) {
            @Override
            public CodecReader wrapForMerge(CodecReader reader) throws IOException {
              // Called by IndexWriter#mergeMiddle once the input readers have been opened (with a
              // merge IOContext) and pooled.
              mergeReadersOpened.countDown();
              try {
                if (!releaseMerge.await(5, TimeUnit.MINUTES)) {
                  throw new IOException("merge was never released");
                }
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException(e);
              }
              return reader;
            }
          });
      return spec;
    }

    @Override
    public MergeSpecification findMerges(
        MergeTrigger mergeTrigger, SegmentInfos segmentInfos, MergeContext mergeContext) {
      return null;
    }
  }

  @Test
  public void testSearchNRTReaderDuringMerge() throws Exception {
    BlockingMergePolicy mergePolicy = new BlockingMergePolicy();
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat()))
            .setMergePolicy(mergePolicy)
            .setMergeScheduler(new ConcurrentMergeScheduler())
            .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setRAMBufferSizeMB(64);

    float[][] vectors = new float[NUM_SEGMENTS * DOCS_PER_SEGMENT][];
    try (Directory dir = newDirectory();
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int seg = 0; seg < NUM_SEGMENTS; seg++) {
        for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
          int id = seg * DOCS_PER_SEGMENT + i;
          vectors[id] = new float[DIMENSION];
          for (int d = 0; d < DIMENSION; d++) {
            vectors[id][d] = random().nextFloat();
          }
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(id), Field.Store.YES));
          doc.add(
              new KnnFloatVectorField("vector", vectors[id], VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();
      }

      try {
        writer.forceMerge(1, false);
        assertTrue(
            "merge did not start", mergePolicy.mergeReadersOpened.await(5, TimeUnit.MINUTES));

        // The segments being merged are still live, so this NRT reader shares the cores the merge
        // opened.
        try (DirectoryReader reader = DirectoryReader.open(writer)) {
          assertEquals(NUM_SEGMENTS, reader.leaves().size());
          IndexSearcher searcher = new IndexSearcher(reader);
          int topK = 10;
          float[] target = vectors[random().nextInt(vectors.length)];
          List<Query> queries =
              List.of(
                  new GPUKnnFloatVectorQuery("vector", target, topK, null, topK, 1),
                  new KnnFloatVectorQuery("vector", target, topK));
          for (Query query : queries) {
            TopDocs hits = searcher.search(query, topK);
            assertEquals(query.toString(), topK, hits.scoreDocs.length);
          }
        }
      } finally {
        mergePolicy.releaseMerge.countDown();
      }
    }
  }
}
