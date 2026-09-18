/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.InfoStream;
import org.junit.Assume;
import org.junit.Test;

/**
 * A CAGRA row is padded to a 16 byte boundary, so a device matrix whose dimension is already a
 * multiple of four sits at the required stride. cuVS refuses to build an owning padded copy of such
 * a matrix and asks for a view instead, and the writer swallows a failed CAGRA build by falling
 * back to a brute force index. The two together are silent: search keeps returning correct results
 * while nothing on the GPU is a CAGRA index any more.
 *
 * <p>These tests pin the dimensions on both sides of that boundary.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestCagraIndexAtAlignedDimensions extends LuceneTestCase {

  @Test
  public void testCagraIsBuiltAtAnAlignedDimension() throws IOException {
    // Deep1B's 96 floats occupy 384 bytes, an exact multiple of CAGRA's 16-byte row alignment.
    assertIndexTypeIsBuilt(IndexType.CAGRA, 96, 64, 1);
  }

  @Test
  public void testCagraIsBuiltAtAnUnalignedDimension() throws IOException {
    // 95 floats is not aligned, so the writer has to create an owning padded copy.
    assertIndexTypeIsBuilt(IndexType.CAGRA, 95, 64, 1);
  }

  @Test
  public void testBruteForceIndexIsBuiltAndSearchable() throws IOException {
    assertIndexTypeIsBuilt(IndexType.BRUTE_FORCE, 32, 64, 1);
  }

  @Test
  public void testCombinedIndexUsesBothSearchPayloads() throws IOException {
    assertIndexTypeIsBuilt(IndexType.CAGRA_AND_BRUTE_FORCE, 32, 1026, 1, 1025);
  }

  /** Indexes one segment and verifies exactly the payloads selected by {@code indexType}. */
  private void assertIndexTypeIsBuilt(
      IndexType indexType, int dimension, int vectorCount, int... searchKs) throws IOException {
    Assume.assumeTrue("Requires a GPU", isSupported());

    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      float[] queryVector = new float[dimension];
      for (int d = 0; d < dimension; d++) {
        queryVector[d] = random().nextFloat();
      }

      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(
                  TestUtil.alwaysKnnVectorsFormat(
                      new CuVS2510GPUVectorsFormat(
                          new GPUSearchParams.Builder().withIndexType(indexType).build())))
              .setInfoStream(infoStream)
              .setMaxBufferedDocs(vectorCount + 1)
              .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int i = 0; i < vectorCount; i++) {
          float[] vector = i == 0 ? queryVector : new float[dimension];
          if (i != 0) {
            for (int d = 0; d < dimension; d++) {
              vector[d] = random().nextFloat();
            }
          }
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        assertEquals("Expected a single segment", 1, reader.leaves().size());
        LeafReader leaf = getOnlyLeafReader(reader);
        CuVS2510GPUVectorsReader gpuReader = gpuReader(leaf, "vector");
        CuVS2510GPUVectorsReader.FieldEntry fieldEntry = gpuReader.getFieldEntry("vector");
        assertNotNull(fieldEntry);
        assertEquals(
            "Unexpected CAGRA payload state",
            indexType.isCagra(),
            fieldEntry.cagraIndexLength() > 0);
        assertEquals(
            "Unexpected brute-force payload state",
            indexType.isBruteForce(),
            fieldEntry.bruteForceIndexLength() > 0);

        int fieldNumber = leaf.getFieldInfos().fieldInfo("vector").number;
        GPUIndex gpuIndex = gpuReader.getCuvsIndexes().get(fieldNumber);
        assertNotNull(gpuIndex);
        assertEquals(
            "Unexpected loaded CAGRA index", indexType.isCagra(), gpuIndex.getCagraIndex() != null);
        assertEquals(
            "Unexpected loaded brute-force index",
            indexType.isBruteForce(),
            gpuIndex.getBruteforceIndex() != null);

        var searcher = newSearcher(reader);
        var query = new GPUKnnFloatVectorQuery("vector", queryVector, 1, null, 1, 1);
        var topDocs = searcher.search(query, 1);
        assertEquals(1, topDocs.scoreDocs.length);
        assertTrue(topDocs.scoreDocs[0].doc >= 0);
        assertTrue(topDocs.scoreDocs[0].doc < reader.maxDoc());

        for (int searchK : searchKs) {
          TopKnnCollector collector = new TopKnnCollector(searchK, Integer.MAX_VALUE);
          leaf.searchNearestVectors("vector", queryVector, collector, null);
          assertEquals(searchK, collector.topDocs().scoreDocs.length);
        }
      }
    }

    if (indexType.isCagra()) {
      assertTrue(
          "The CAGRA build failed for "
              + indexType
              + " at dimension "
              + dimension
              + ", messages: "
              + infoStream.messages(),
          infoStream.cagraBuildFailures().isEmpty());
    }
  }

  private static CuVS2510GPUVectorsReader gpuReader(LeafReader leaf, String field) {
    KnnVectorsReader reader = ((CodecReader) leaf).getVectorReader();
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      reader = fieldsReader.getFieldReader(field);
    }
    return (CuVS2510GPUVectorsReader) reader;
  }

  /** An InfoStream that keeps the messages, so that a test can tell which index type was built. */
  private static class RecordingInfoStream extends InfoStream {

    private final List<String> messages = Collections.synchronizedList(new ArrayList<>());

    @Override
    public void message(String component, String message) {
      messages.add(component + ": " + message);
    }

    @Override
    public boolean isEnabled(String component) {
      return true;
    }

    @Override
    public void close() {}

    List<String> messages() {
      synchronized (messages) {
        return List.copyOf(messages);
      }
    }

    List<String> cagraBuildFailures() {
      return messages().stream().filter(message -> message.contains("CAGRA build failed")).toList();
    }
  }
}
