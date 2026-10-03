/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

/**
 * Writes the back-compat indexes of a released cuvs-lucene artifact, for TestBackCompatIndices.
 *
 * <p>Not part of the build: generate-released-indices.sh compiles and runs it against the published
 * jars of one release. It writes the same documents as TestBackCompatIndices#writeIndex, so that
 * the same checks apply to indexes written by released code.
 *
 * <p>It writes one index with each cuvs-lucene codec the release registers. Releases that
 * registered no codecs (25.12 and earlier) instead get one index per cuvs-lucene vectors format,
 * used per field by Lucene's default codec, which is how those releases were used. The GPU search
 * codec and format are left out: their on-disk format is documented as experimental.
 *
 * <p>Arguments: the output directory and the cuvs-lucene version.
 */
public class GenerateReleasedIndices {

  // Keep in sync with TestBackCompatIndices.
  private static final int NUM_DOCS = 200;
  private static final int DIMENSIONS = 8;

  public static void main(String[] args) throws Exception {
    Path outDir = Path.of(args[0]);
    String cuvsVersion = args[1];
    Files.createDirectories(outDir);
    String lucene = Version.LATEST.major + "." + Version.LATEST.minor;
    List<String> codecs =
        Codec.availableCodecs().stream()
            .filter(GenerateReleasedIndices::isCuVSCodec)
            .sorted()
            .toList();
    List<String> formats =
        codecs.isEmpty()
            ? KnnVectorsFormat.availableKnnVectorsFormats().stream()
                .filter(GenerateReleasedIndices::isCuVSFormat)
                .sorted()
                .toList()
            : List.of();
    for (String name : codecs) {
      write(Codec.forName(name), outDir, lucene + "-" + name + "-cuvs-" + cuvsVersion + ".zip");
    }
    for (String name : formats) {
      Codec base = Codec.getDefault();
      KnnVectorsFormat format = KnnVectorsFormat.forName(name);
      Codec codec =
          new FilterCodec(base.getName(), base) {
            private final KnnVectorsFormat perField =
                new PerFieldKnnVectorsFormat() {
                  @Override
                  public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    return format;
                  }
                };

            @Override
            public KnnVectorsFormat knnVectorsFormat() {
              return perField;
            }
          };
      String fileName =
          lucene + "-" + base.getName() + "-" + name + "-cuvs-" + cuvsVersion + ".zip";
      write(codec, outDir, fileName);
    }
  }

  private static boolean isCuVSCodec(String name) {
    String cls = Codec.forName(name).getClass().getName();
    return cls.startsWith("com.nvidia.cuvs.") && !cls.contains("GPU");
  }

  private static boolean isCuVSFormat(String name) {
    String cls = KnnVectorsFormat.forName(name).getClass().getName();
    return cls.startsWith("com.nvidia.cuvs.") && !cls.contains("GPU");
  }

  private static void write(Codec codec, Path outDir, String fileName) throws Exception {
    Path indexDir = Files.createTempDirectory("backcompat");
    writeIndex(indexDir, codec);
    zip(indexDir, outDir.resolve(fileName));
    System.out.println("wrote " + fileName);
  }

  private static void writeIndex(Path indexDir, Codec codec) throws Exception {
    Random random = new Random(42);
    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setUseCompoundFile(false)
            .setMergePolicy(NoMergePolicy.INSTANCE);
    try (Directory dir = FSDirectory.open(indexDir);
        IndexWriter writer = new IndexWriter(dir, config)) {
      for (int i = 0; i < NUM_DOCS; i++) {
        float[] vector = new float[DIMENSIONS];
        for (int d = 0; d < DIMENSIONS; d++) {
          vector[d] = random.nextFloat();
        }
        Document doc = new Document();
        doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
        doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc);
      }
      writer.commit();
    }
  }

  private static void zip(Path indexDir, Path zip) throws Exception {
    List<String> files;
    try (Directory dir = FSDirectory.open(indexDir)) {
      files = SegmentInfos.readLatestCommit(dir).files(true).stream().sorted().toList();
    }
    try (OutputStream out = Files.newOutputStream(zip);
        ZipOutputStream zipOut = new ZipOutputStream(out)) {
      for (String file : files) {
        ZipEntry entry = new ZipEntry(file);
        entry.setTime(0);
        zipOut.putNextEntry(entry);
        try (InputStream in = Files.newInputStream(indexDir.resolve(file))) {
          in.transferTo(zipOut);
        }
        zipOut.closeEntry();
      }
    }
  }
}
