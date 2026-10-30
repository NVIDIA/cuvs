/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
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
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
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
 * <p>Released versions only cover the Lucene releases they supported, so each module also writes
 * indexes with the codecs from {@link CuVSCodecs} to {@code <dir>/<lucene major.minor>/<codec>.zip}
 * in the directory the {@value #DIR_PROPERTY} system property names, and reads and upgrades the
 * ones that modules for earlier Lucene releases wrote there. The parent pom points every module at
 * the same directory, and Maven builds the modules from the oldest Lucene release up, so one {@code
 * mvn test} checks every upgrade from one supported Lucene release to a later one with the code
 * being tested.
 *
 * <p>Reading and upgrading need no GPU, so those tests also run where cuVS is not available.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestBackCompatIndices extends LuceneTestCase {

  private static final String RESOURCE_DIR = "/backcompat";
  private static final String DIR_PROPERTY = "tests.cuvs.backCompatDir";
  // The path of an index below RESOURCE_DIR; see the class javadoc.
  private static final Pattern INDEX_PATH =
      Pattern.compile("releases/[\\d.]+/(\\d+)\\.(\\d+)/(\\w+)\\.zip");
  // The path of an index below DIR_PROPERTY, written by a module of this build.
  private static final Pattern MODULE_INDEX_PATH = Pattern.compile("(\\d+)\\.(\\d+)/(\\w+)\\.zip");
  private static final String ID_FIELD = "id";
  private static final String VECTOR_FIELD = "vector";
  private static final int NUM_DOCS = 200;
  private static final int DIMENSIONS = 8;
  private static final int TOP_K = 10;
  private static final int UPGRADE_DOCS = 50;

  public void testReadIndices() throws Exception {
    for (Path zip : indicesReadableByThisRelease()) {
      readIndex(zip, indexPath(zip));
    }
  }

  /**
   * Opens each index for writing with the codec that writes on this Lucene release, adds documents
   * and merges everything into one segment, which rewrites the old segments with the current codec,
   * as an application does after moving to a later release.
   */
  public void testUpgradeIndices() throws Exception {
    for (Path zip : indicesReadableByThisRelease()) {
      upgradeIndex(zip, indexPath(zip));
    }
  }

  /** Writes the indexes that the modules for later Lucene releases read. */
  public void testWriteIndicesForLaterReleases() throws Exception {
    // Written on the GPU, like the indexes of the applications this test stands for.
    assumeTrue("cuVS not supported", isSupported());
    Path dir = moduleIndexDir();
    assumeTrue("set -D" + DIR_PROPERTY + " to write the indexes", dir != null);
    Path targetDir = dir.resolve(Version.LATEST.major + "." + Version.LATEST.minor);
    Files.createDirectories(targetDir);
    for (Codec codec : codecsToWrite()) {
      Path indexDir = createTempDir(codec.getName());
      writeIndex(indexDir, codec);
      zip(indexDir, targetDir.resolve(codec.getName() + ".zip"));
    }
  }

  public void testReadIndicesFromEarlierReleases() throws Exception {
    for (Path zip : indicesFromEarlierModules()) {
      readIndex(zip, moduleIndexPath(zip));
    }
  }

  public void testUpgradeIndicesFromEarlierReleases() throws Exception {
    for (Path zip : indicesFromEarlierModules()) {
      upgradeIndex(zip, moduleIndexPath(zip));
    }
  }

  private void readIndex(Path zip, Matcher matcher) throws IOException {
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
      throw new AssertionError("Failed to read " + zip, e);
    }
  }

  private void upgradeIndex(Path zip, Matcher matcher) throws IOException {
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
      throw new AssertionError("Failed to upgrade " + zip, e);
    }
  }

  /** The codecs that write the indexes; the GPU search codec's format makes no such promise. */
  private static List<Codec> codecsToWrite() {
    return List.of(
        CuVSCodecs.acceleratedHNSW(),
        CuVSCodecs.acceleratedHNSWScalarQuantized(),
        CuVSCodecs.acceleratedHNSWBinaryQuantized());
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

  /** Returns the directory {@link #DIR_PROPERTY} names, or {@code null} if it is not set. */
  private static Path moduleIndexDir() {
    String dir = System.getProperty(DIR_PROPERTY);
    return dir == null || dir.isEmpty() ? null : Path.of(dir);
  }

  /** Matches the path of an index zip below {@link #moduleIndexDir()}. */
  private static Matcher moduleIndexPath(Path zip) {
    String path = moduleIndexDir().relativize(zip).toString().replace('\\', '/');
    Matcher matcher = MODULE_INDEX_PATH.matcher(path);
    assertTrue("unexpected back-compat index path: " + path, matcher.matches());
    return matcher;
  }

  /**
   * Compares the Lucene release an index path records ({@code major.minor} in groups 1 and 2) with
   * the one in use: negative if it is earlier, 0 if it is the same, positive if it is later.
   */
  private static int compareToLuceneInUse(Matcher indexPath) {
    int major = Integer.parseInt(indexPath.group(1));
    int minor = Integer.parseInt(indexPath.group(2));
    return major != Version.LATEST.major
        ? Integer.compare(major, Version.LATEST.major)
        : Integer.compare(minor, Version.LATEST.minor);
  }

  /**
   * Returns the indexes the modules for earlier Lucene releases wrote with {@link
   * #testWriteIndicesForLaterReleases}, and skips the test if there are none.
   */
  private static List<Path> indicesFromEarlierModules() throws IOException {
    Path dir = moduleIndexDir();
    assumeTrue("set -D" + DIR_PROPERTY + " to read the indexes of earlier releases", dir != null);
    List<Path> zips = new ArrayList<>();
    if (Files.isDirectory(dir)) {
      try (Stream<Path> files = Files.walk(dir)) {
        for (Path f : files.filter(f -> f.toString().endsWith(".zip")).sorted().toList()) {
          if (compareToLuceneInUse(moduleIndexPath(f)) < 0) {
            zips.add(f);
          }
        }
      }
    }
    // The module for the oldest supported release has none to read, and a module tested on its
    // own has none unless an earlier build wrote them.
    assumeFalse(
        "no indexes written by the modules for earlier Lucene releases in "
            + dir
            + "; test those modules in the same build to write them",
        zips.isEmpty());
    return zips;
  }

  private static List<Path> indicesReadableByThisRelease() throws IOException, URISyntaxException {
    List<Path> zips = new ArrayList<>();
    try (Stream<Path> files = Files.walk(resourceDir())) {
      for (Path f : files.filter(f -> f.toString().endsWith(".zip")).sorted().toList()) {
        // Every zip has to be where the tests expect it, or a misplaced one would go unread.
        if (compareToLuceneInUse(indexPath(f)) <= 0) {
          zips.add(f);
        }
      }
    }
    assertFalse("no back-compat indexes found under " + RESOURCE_DIR, zips.isEmpty());
    return zips;
  }

  /** Writes the same documents as {@code GenerateReleasedIndices#writeIndex}. */
  private static void writeIndex(Path indexDir, Codec codec) throws IOException {
    Random random = new Random(42);
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMergePolicy(NoMergePolicy.INSTANCE);
    try (Directory dir = FSDirectory.open(indexDir);
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < NUM_DOCS; i++) {
        writer.addDocument(document(i, randomVector(random)));
      }
      writer.commit();
    }
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

  /**
   * Zips the files of the index's latest commit, leaving out anything else in the directory. The
   * zip is written under a temporary name and then renamed, so that a module reading the directory
   * meanwhile, as one built in parallel can, never sees a partly written zip.
   */
  private static void zip(Path indexDir, Path zip) throws IOException {
    List<String> files;
    try (Directory dir = FSDirectory.open(indexDir)) {
      files = SegmentInfos.readLatestCommit(dir).files(true).stream().sorted().toList();
    }
    Path tmp = zip.resolveSibling(zip.getFileName() + ".tmp");
    try (OutputStream out = Files.newOutputStream(tmp);
        ZipOutputStream zipOut = new ZipOutputStream(out)) {
      for (String file : files) {
        zipOut.putNextEntry(new ZipEntry(file));
        try (InputStream in = Files.newInputStream(indexDir.resolve(file))) {
          in.transferTo(zipOut);
        }
        zipOut.closeEntry();
      }
    }
    Files.move(tmp, zip, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
  }
}
