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
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.ConcurrentMergeScheduler;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
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
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
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

  /** Sums the indexes the segments' vectors readers have loaded lazily, over both fields. */
  private static int lazilyLoadedIndexCount(DirectoryReader reader) {
    int count = 0;
    for (LeafReaderContext ctx : reader.leaves()) {
      // Both fields use the same format, so they share one reader.
      count += gpuVectorsReader(ctx).lazilyLoadedIndexCount();
    }
    return count;
  }

  private static CuVS2510GPUVectorsReader gpuVectorsReader(LeafReaderContext ctx) {
    return (CuVS2510GPUVectorsReader) TestUtils.vectorsReader(ctx.reader(), "vector");
  }

  /**
   * A directory whose inputs report that they suit no reads, so that since Lucene 10.6 a merge opens
   * the raw vectors again instead of reading them through the input searches use.
   */
  private static final class SingleUseInputsDirectory extends FilterDirectory {
    SingleUseInputsDirectory(Directory in) {
      super(in);
    }

    @Override
    public IndexInput openInput(String name, IOContext context) throws IOException {
      return new FilterIndexInput("single use(" + name + ")", super.openInput(name, context)) {
        // Added to IndexInput in Lucene 10.6, so it cannot be marked @Override.
        public boolean isEfficientFor(IOContext other) {
          return false;
        }

        // Clones read on their own, rather than share this input's position.
        @Override
        public IndexInput clone() {
          return in.clone();
        }
      };
    }
  }

  /**
   * A merge reads a segment through the merge instance of its vectors reader, which since Lucene
   * 10.6 reads the raw vectors through a reader of its own, when the input searches use does not
   * suit a merge, but shares everything else with the segment's reader. Closing it must leave that
   * reader open, and a merge through it must write a searchable segment.
   */
  @Test
  public void testMergeInstance() throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat()))
            .setUseCompoundFile(false);
    float[][] vectors = new float[2 * DOCS_PER_SEGMENT][];
    try (Directory dir = new SingleUseInputsDirectory(newDirectory());
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < vectors.length; i++) {
        vectors[i] = TestUtils.generateRandomVector(DIMENSION, random());
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("vector", vectors[i], VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc);
        if (i == DOCS_PER_SEGMENT - 1) {
          writer.commit();
        }
      }
      writer.commit();

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        CuVS2510GPUVectorsReader vectorsReader = gpuVectorsReader(reader.leaves().get(0));
        KnnVectorsReader mergeInstance = vectorsReader.getMergeInstance();
        assertTrue(mergeInstance instanceof CuVS2510GPUVectorsReader);
        if (LuceneCompat.LUCENE_MAJOR > 10 || LuceneCompat.LUCENE_MINOR >= 6) {
          assertNotSame(vectorsReader, mergeInstance);
        }
        FloatVectorValues values = mergeInstance.getFloatVectorValues("vector");
        assertEquals(DOCS_PER_SEGMENT, values.size());
        for (int ord = 0; ord < DOCS_PER_SEGMENT; ord++) {
          assertArrayEquals(vectors[ord], values.vectorValue(ord), 0f);
        }
        mergeInstance.finishMerge();
        if (mergeInstance != vectorsReader) {
          mergeInstance.close();
        }

        int topK = 10;
        TopDocs hits =
            new IndexSearcher(reader)
                .search(
                    new GPUKnnFloatVectorQuery("vector", vectors[0], topK, null, topK, 1), topK);
        assertEquals(topK, hits.scoreDocs.length);
        assertEquals(0, hits.scoreDocs[0].doc);
      }

      // A real merge, which reads both segments through their readers' merge instances.
      writer.forceMerge(1);
      try (DirectoryReader reader = DirectoryReader.open(writer)) {
        assertEquals(1, reader.leaves().size());
        int topK = 10;
        IndexSearcher searcher = new IndexSearcher(reader);
        for (int doc : new int[] {0, vectors.length - 1}) {
          TopDocs hits =
              searcher.search(
                  new GPUKnnFloatVectorQuery("vector", vectors[doc], topK, null, topK, 1), topK);
          assertEquals(topK, hits.scoreDocs.length);
          assertEquals(doc, hits.scoreDocs[0].doc);
        }
      }
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
          doc.add(
              new KnnFloatVectorField("vector2", vectors[id], VectorSimilarityFunction.EUCLIDEAN));
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
          // Searching one field loaded that field's index only.
          assertEquals(NUM_SEGMENTS, lazilyLoadedIndexCount(reader));

          TopDocs hits =
              searcher.search(
                  new GPUKnnFloatVectorQuery("vector2", target, topK, null, topK, 1), topK);
          assertEquals(topK, hits.scoreDocs.length);
          assertEquals(2 * NUM_SEGMENTS, lazilyLoadedIndexCount(reader));
        }
      } finally {
        mergePolicy.releaseMerge.countDown();
      }

      // The merge finishes with the searched readers still pooled, and its result is searchable.
      writer.forceMerge(1);
      try (DirectoryReader reader = DirectoryReader.open(writer)) {
        assertEquals(1, reader.leaves().size());
        for (String field : List.of("vector", "vector2")) {
          TopDocs hits =
              new IndexSearcher(reader).search(new KnnFloatVectorQuery(field, vectors[0], 10), 10);
          assertEquals(10, hits.scoreDocs.length);
        }
      }
    }
  }
}
