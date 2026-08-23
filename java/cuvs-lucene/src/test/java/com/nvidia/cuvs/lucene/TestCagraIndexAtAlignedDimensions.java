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
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
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
    // 128 floats is 512 bytes, an exact multiple of the 16 byte CAGRA row alignment.
    assertCagraIsBuilt(128);
  }

  @Test
  public void testCagraIsBuiltAtAnUnalignedDimension() throws IOException {
    // 127 floats is not, so the writer has to fall back to an owning padded copy.
    assertCagraIsBuilt(127);
  }

  /** Indexes a segment of the given dimension and fails if the CAGRA build did not survive it. */
  private void assertCagraIsBuilt(int dimension) throws IOException {
    Assume.assumeTrue("Requires a GPU", isSupported());

    RecordingInfoStream infoStream = new RecordingInfoStream();
    try (Directory directory = newDirectory()) {
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(
                  TestUtil.alwaysKnnVectorsFormat(
                      new CuVS2510GPUVectorsFormat(
                          new GPUSearchParams.Builder().withIndexType(IndexType.CAGRA).build())))
              .setInfoStream(infoStream);

      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int i = 0; i < 64; i++) {
          float[] vector = new float[dimension];
          for (int d = 0; d < dimension; d++) {
            vector[d] = random().nextFloat();
          }
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();
      }
    }

    assertTrue(
        "The CAGRA build fell back to brute force at dimension "
            + dimension
            + ", messages: "
            + infoStream.messages(),
        infoStream.cagraBuildFailures().isEmpty());
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
