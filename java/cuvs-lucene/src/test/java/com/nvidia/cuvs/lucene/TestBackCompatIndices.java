/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.util.Version;

/**
 * Opens and upgrades indexes written with the cuvs-lucene codecs on earlier Lucene releases and by
 * earlier cuvs-lucene releases, to make sure an application can still use its indexes after it
 * moves to a later Lucene release, cuvs-lucene release, or both.
 *
 * <p>The indexes are in {@code src/test/resources/backcompat}, one zip each, at {@code
 * releases/<cuvs-lucene version>/<lucene major.minor>/<name>.zip}, written by the jars of a released
 * cuvs-lucene version. The path records the Lucene release they were written on and either the
 * cuvs-lucene codec recorded in their segments or, for an index written with Lucene's own codec,
 * the cuvs-lucene vectors format its vector field uses. Each module reads the ones written on its own Lucene
 * release and all earlier ones. See the {@code README.md} next to them for how to add more.
 *
 * <p>Reading and upgrading need no GPU, so those tests also run where cuVS is not available.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestBackCompatIndices extends LuceneTestCase {

  private static final String RESOURCE_DIR = "/backcompat";
  // The path of an index below RESOURCE_DIR; see the class javadoc.
  private static final Pattern INDEX_PATH =
      Pattern.compile("releases/[\\d.]+/(\\d+)\\.(\\d+)/(\\w+)\\.zip");
  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector";
  private static final int NUM_DOCS = 200;
  private static final int DIMENSIONS = 8;
  private static final int TOP_K = 10;
  private static final int UPGRADE_DOCS = 50;

  public void testReadIndices() throws Exception {
    for (Path zip : indicesReadableByThisRelease()) {
      Matcher matcher = indexPath(zip);
      Path indexDir = createTempDir(zip.getFileName().toString());
      unzip(zip, indexDir);
      try (Directory dir = newFSDirectory(indexDir)) {
        String name = matcher.group(3);
        if (KnnVectorsFormat.availableKnnVectorsFormats().contains(name)) {
          // Lucene's own codec with a cuvs-lucene format per field: check the field's format.
          checkIndex(dir, null, name, NUM_DOCS);
        } else {
          // A cuvs-lucene codec: check the codec recorded in every segment.
          checkIndex(dir, name, null, NUM_DOCS);
        }
      } catch (AssertionError | Exception e) {
        throw new AssertionError("Failed to read " + matcher.group(), e);
      }
    }
  }

  /**
   * Opens each index for writing with the codec that writes on this Lucene release, adds documents
   * and merges everything into one segment, which rewrites the old segments with the current codec,
   * as an application does after moving to a later release.
   */
  public void testUpgradeIndices() throws Exception {
    for (Path zip : indicesReadableByThisRelease()) {
      Matcher matcher = indexPath(zip);
      Codec codec = currentCodecFor(matcher.group(3));
      Path indexDir = createTempDir(zip.getFileName().toString());
      unzip(zip, indexDir);
      try (Directory dir = newFSDirectory(indexDir)) {
        IndexWriterConfig config =
            new IndexWriterConfig()
                .setCodec(codec)
                .setOpenMode(IndexWriterConfig.OpenMode.APPEND)
                .setUseCompoundFile(false);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
          Random random = new Random(7);
          for (int i = NUM_DOCS; i < NUM_DOCS + UPGRADE_DOCS; i++) {
            writer.addDocument(document(i, randomVector(random)));
          }
          writer.forceMerge(1);
        }
        checkIndex(dir, codec.getName(), null, NUM_DOCS + UPGRADE_DOCS);
      } catch (AssertionError | Exception e) {
        throw new AssertionError("Failed to upgrade " + matcher.group(), e);
      }
    }
  }

  /** Returns the codec that writes on this Lucene release in the family of the given codec. */
  private static Codec currentCodecFor(String codecName) {
    if (codecName.contains("ScalarQuantized")) {
      return CuVSCodecs.acceleratedHNSWScalarQuantized();
    } else if (codecName.contains("BinaryQuantized")) {
      return CuVSCodecs.acceleratedHNSWBinaryQuantized();
    }
    // The accelerated HNSW codecs, and Lucene's own codecs with its vectors format used per field.
    return CuVSCodecs.acceleratedHNSW();
  }

  private static Path resourceDir() throws URISyntaxException {
    URL url = TestBackCompatIndices.class.getResource(RESOURCE_DIR);
    assertNotNull(RESOURCE_DIR + " not found on the test classpath", url);
    return Path.of(url.toURI());
  }

  /** Matches the path of an index zip below {@link #RESOURCE_DIR} against {@link #INDEX_PATH}. */
  private static Matcher indexPath(Path zip) throws URISyntaxException {
    String path = resourceDir().relativize(zip).toString().replace('\\', '/');
    Matcher matcher = INDEX_PATH.matcher(path);
    assertTrue("unexpected back-compat index path: " + path, matcher.matches());
    return matcher;
  }

  private static List<Path> indicesReadableByThisRelease() throws IOException, URISyntaxException {
    List<Path> zips = new ArrayList<>();
    try (Stream<Path> files = Files.walk(resourceDir())) {
      for (Path f : files.filter(f -> f.toString().endsWith(".zip")).sorted().toList()) {
        // Every zip has to be where the tests expect it, or a misplaced one would go unread.
        Matcher matcher = indexPath(f);
        int major = Integer.parseInt(matcher.group(1));
        int minor = Integer.parseInt(matcher.group(2));
        if (major < Version.LATEST.major
            || (major == Version.LATEST.major && minor <= Version.LATEST.minor)) {
          zips.add(f);
        }
      }
    }
    assertFalse("no back-compat indexes found under " + RESOURCE_DIR, zips.isEmpty());
    return zips;
  }

  private static float[] randomVector(Random random) {
    float[] vector = new float[DIMENSIONS];
    for (int d = 0; d < DIMENSIONS; d++) {
      vector[d] = random.nextFloat();
    }
    return vector;
  }

  private static Document document(int id, float[] vector) {
    Document doc = new Document();
    doc.add(new StringField(ID_FIELD, Integer.toString(id), Field.Store.YES));
    doc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, EUCLIDEAN));
    return doc;
  }

  /**
   * Checks that every segment was written by the expected codec, or that the vector field uses the
   * expected cuvs-lucene format, and that searching for each indexed vector finds its own document.
   *
   * @param codecName the codec recorded in every segment, or null to skip
   * @param perFieldFormat the vectors format name recorded for the vector field, or null to skip
   */
  private static void checkIndex(
      Directory dir, String codecName, String perFieldFormat, int numDocs) throws IOException {
    if (codecName != null) {
      for (SegmentCommitInfo info : SegmentInfos.readLatestCommit(dir)) {
        assertEquals(codecName, info.info.getCodec().getName());
      }
    }
    try (DirectoryReader reader = DirectoryReader.open(dir)) {
      assertEquals(numDocs, reader.numDocs());
      if (perFieldFormat != null) {
        for (var context : reader.leaves()) {
          FieldInfo field = context.reader().getFieldInfos().fieldInfo(VECTOR_FIELD);
          assertEquals(
              perFieldFormat, field.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY));
        }
      }
      IndexSearcher searcher = new IndexSearcher(reader);
      StoredFields allStoredFields = searcher.storedFields();
      int found = 0;
      for (var context : reader.leaves()) {
        LeafReader leaf = context.reader();
        FloatVectorValues vectors = leaf.getFloatVectorValues(VECTOR_FIELD);
        assertEquals(DIMENSIONS, vectors.dimension());
        StoredFields storedFields = leaf.storedFields();
        KnnVectorValues.DocIndexIterator it = vectors.iterator();
        for (int doc = it.nextDoc(); doc != NO_MORE_DOCS; doc = it.nextDoc()) {
          String id = storedFields.document(doc).get(ID_FIELD);
          float[] query = vectors.vectorValue(it.index()).clone();
          ScoreDoc[] hits =
              searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, TOP_K), TOP_K).scoreDocs;
          for (ScoreDoc hit : hits) {
            if (id.equals(allStoredFields.document(hit.doc).get(ID_FIELD))) {
              found++;
              break;
            }
          }
        }
      }
      // Quantized codecs may rank a few near-duplicates ahead of the exact match.
      assertTrue(
          "only " + found + " of " + numDocs + " vectors found themselves", found >= numDocs * 0.9);
    }
  }

  private static void unzip(Path zip, Path targetDir) throws IOException {
    try (ZipInputStream in = new ZipInputStream(Files.newInputStream(zip))) {
      for (ZipEntry entry = in.getNextEntry(); entry != null; entry = in.getNextEntry()) {
        Files.copy(in, targetDir.resolve(entry.getName()));
      }
    }
  }
}
