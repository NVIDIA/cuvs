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
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;

/**
 * Searches read the vectors under a graph at random. Since Lucene 10.6 the flat format no
 * longer says so itself, and the format wrapping it has to; this checks that the cuvs-lucene
 * formats open the raw vectors the way Lucene's HNSW format does.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestRawVectorsReadContext extends LuceneTestCase {

  private static final int NUM_DOCS = 20;
  private static final int DIMENSION = 8;

  /** Records the contexts the raw and quantized vectors files are opened with. */
  private static final class RecordingDirectory extends FilterDirectory {
    final List<IOContext> rawVectorContexts = new ArrayList<>();

    RecordingDirectory(Directory in) {
      super(in);
    }

    @Override
    public IndexInput openInput(String name, IOContext context) throws IOException {
      if (name.endsWith(".vec") || name.endsWith(".veq")) {
        rawVectorContexts.add(context);
      }
      return super.openInput(name, context);
    }
  }

  /** Returns the contexts that opening a reader opens the vectors of {@code format} with. */
  private List<IOContext> rawVectorContexts(KnnVectorsFormat format) throws IOException {
    try (RecordingDirectory dir = new RecordingDirectory(newDirectory())) {
      IndexWriterConfig config =
          new IndexWriterConfig()
              .setCodec(alwaysKnnVectorsFormat(format))
              .setUseCompoundFile(false);
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        for (int i = 0; i < NUM_DOCS; i++) {
          Document doc = new Document();
          doc.add(
              new KnnFloatVectorField(
                  "vector",
                  TestUtils.generateRandomVector(DIMENSION, random()),
                  VectorSimilarityFunction.EUCLIDEAN));
          writer.addDocument(doc);
        }
      }
      dir.rawVectorContexts.clear();
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertEquals(1, reader.leaves().size());
      }
      assertFalse(format.getName(), dir.rawVectorContexts.isEmpty());
      return dir.rawVectorContexts;
    }
  }

  public void testRawVectorsOpenedLikeLuceneHnsw() throws IOException {
    List<IOContext> expected = rawVectorContexts(new Lucene99HnswVectorsFormat());
    List<KnnVectorsFormat> formats = new ArrayList<>();
    formats.add(CuVSCodecs.acceleratedHNSWFormat());
    formats.add(CuVSCodecs.acceleratedHNSWBinaryQuantizedFormat());
    if (isSupported()) {
      formats.add(CuVSCodecs.gpuSearchFormat());
    }
    for (KnnVectorsFormat format : formats) {
      assertEquals(format.getName(), expected, rawVectorContexts(format));
    }
  }

  public void testScalarQuantizedVectorsOpenedLikeLuceneHnsw() throws IOException {
    // Lucene's HNSW format over the same flat format as the one that writes on this release, looked
    // up by name because Lucene 10.2 and 10.3 have no Lucene104 formats. Lucene registers
    // Lucene104HnswScalarQuantizedVectorsFormat as "Lucene104HnswBinaryQuantizedVectorsFormat".
    String luceneFormat =
        LuceneCompat.SCALAR_QUANTIZED_FORMAT_NAME.startsWith("Lucene104")
            ? "Lucene104HnswBinaryQuantizedVectorsFormat"
            : "Lucene99HnswScalarQuantizedVectorsFormat";
    List<IOContext> expected = rawVectorContexts(KnnVectorsFormat.forName(luceneFormat));
    KnnVectorsFormat format = CuVSCodecs.acceleratedHNSWScalarQuantizedFormat();
    assertEquals(format.getName(), expected, rawVectorContexts(format));
  }
}
