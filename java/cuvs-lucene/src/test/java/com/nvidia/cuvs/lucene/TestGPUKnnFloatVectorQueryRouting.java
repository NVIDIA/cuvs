/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.CagraSearchParams;
import java.util.HashSet;
import java.util.Set;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.Test;

/** Verifies explicit CAGRA algorithm choices at the Lucene query-routing boundary. */
@SuppressSysoutChecks(bugUrl = "")
public class TestGPUKnnFloatVectorQueryRouting extends LuceneTestCase {

  private static final String VECTOR_FIELD = "vector";
  private static final int DIMENSIONS = 128;
  private static final int SEGMENT_COUNT = 2;
  private static final int DOCUMENTS_PER_SEGMENT = 512;
  private static final int TOP_K = 10;
  private static final int I_TOP_K = 32;

  @Test
  public void testExplicitMultiKernelUsesPerSegmentCagraSearch() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    float[][] dataset =
        generateDataset(random(), SEGMENT_COUNT * DOCUMENTS_PER_SEGMENT, DIMENSIONS);
    GPUSearchParams buildParameters =
        new GPUSearchParams.Builder()
            .withStrategy(GPUSearchParams.Strategy.CUSTOM)
            .withGraphDegree(32)
            .withIntermediateGraphDegree(64)
            .build();

    try (Directory directory = newDirectory()) {
      writeCagraSegments(directory, dataset, buildParameters);

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals(SEGMENT_COUNT, reader.leaves().size());
        assertEverySegmentLoadedOnlyCagra(reader);
        IndexSearcher searcher = newSearcher(reader);
        GPUKnnFloatVectorQuery query =
            new GPUKnnFloatVectorQuery(
                VECTOR_FIELD,
                dataset[0],
                TOP_K,
                null,
                I_TOP_K,
                1,
                0,
                0,
                CagraSearchParams.SearchAlgo.MULTI_KERNEL);

        ScoreDoc[] hits = searcher.search(query, TOP_K).scoreDocs;

        assertEquals(TOP_K, hits.length);
        assertEquals("the query document must rank first", 0, hits[0].doc);
        assertNoDuplicateDocuments(hits);
      }
    }
  }

  private static void assertEverySegmentLoadedOnlyCagra(DirectoryReader reader) {
    for (LeafReaderContext leaf : reader.leaves()) {
      assertTrue("segment is not backed by a CodecReader", leaf.reader() instanceof CodecReader);
      KnnVectorsReader vectorsReader = ((CodecReader) leaf.reader()).getVectorReader();
      if (vectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perFieldReader) {
        vectorsReader = perFieldReader.getFieldReader(VECTOR_FIELD);
      }
      assertTrue(
          "segment is not backed by the cuVS GPU vectors reader",
          vectorsReader instanceof CuVS2510GPUVectorsReader);

      CuVS2510GPUVectorsReader gpuReader = (CuVS2510GPUVectorsReader) vectorsReader;
      FieldInfo vectorField = gpuReader.getFieldInfos().fieldInfo(VECTOR_FIELD);
      assertNotNull("segment is missing the vector field", vectorField);
      assertNotNull("segment did not load any GPU indexes", gpuReader.getCuvsIndexes());
      GPUIndex gpuIndex = gpuReader.getCuvsIndexes().get(vectorField.number);
      assertNotNull("segment did not load the vector field's GPU index", gpuIndex);
      assertNotNull("segment did not load a CAGRA index", gpuIndex.getCagraIndex());
      assertNull("segment unexpectedly loaded a brute-force index", gpuIndex.getBruteforceIndex());
    }
  }

  private static void writeCagraSegments(
      Directory directory, float[][] dataset, GPUSearchParams buildParameters) throws Exception {
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(new CuVS2510GPUSearchCodec(buildParameters))
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setUseCompoundFile(false);
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int documentId = 0; documentId < dataset.length; documentId++) {
        Document document = new Document();
        document.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[documentId], EUCLIDEAN));
        writer.addDocument(document);
        if ((documentId + 1) % DOCUMENTS_PER_SEGMENT == 0) {
          writer.commit();
        }
      }
    }
  }

  private static void assertNoDuplicateDocuments(ScoreDoc[] hits) {
    Set<Integer> documentIds = new HashSet<>();
    for (ScoreDoc hit : hits) {
      assertTrue("duplicate result for document " + hit.doc, documentIds.add(hit.doc));
    }
  }
}
