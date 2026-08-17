/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;

/**
 * Dynamically loads Lucene format, reader, and writer classes with a fallback mechanism.
 *
 * @since 25.12
 */
public class LuceneProvider {

  static final Logger log = Logger.getLogger(LuceneProvider.class.getName());
  static final String LUCENE_99_FORMAT_VERSION = "99";
  static final String LUCENE_102_BINARY_FORMAT_VERSION = "102";

  private static final String BASE = "org.apache.lucene.";
  private static String codecs = "codecs.lucene<version>.";
  private static String fallbackCodecs = "backward_codecs.lucene<version>.";

  private static String luceneFlatVectorsFormat =
      BASE + codecs + "Lucene<version>FlatVectorsFormat";
  private static String luceneFlatVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>FlatVectorsFormat";

  private static String luceneHnswVectorsFormat =
      BASE + codecs + "Lucene<version>HnswVectorsFormat";
  private static String luceneHnswVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswVectorsFormat";

  private static String luceneHnswVectorsReader =
      BASE + codecs + "Lucene<version>HnswVectorsReader";
  private static String luceneHnswVectorsReaderFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswVectorsReader";

  private static String luceneHnswVectorsWriter =
      BASE + codecs + "Lucene<version>HnswVectorsWriter";
  private static String luceneHnswVectorsWriterFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswVectorsWriter";

  private static String luceneBinaryQuantizedVectorsFormat =
      BASE + codecs + "Lucene<version>BinaryQuantizedVectorsFormat";
  private static String luceneBinaryQuantizedVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>BinaryQuantizedVectorsFormat";

  private static String luceneHnswBinaryQuantizedVectorsFormat =
      BASE + codecs + "Lucene<version>HnswBinaryQuantizedVectorsFormat";
  private static String luceneHnswBinaryQuantizedVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswBinaryQuantizedVectorsFormat";

  private static String luceneScalarQuantizedVectorsFormat =
      BASE + codecs + "Lucene<version>ScalarQuantizedVectorsFormat";
  private static String luceneScalarQuantizedVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>ScalarQuantizedVectorsFormat";

  private static String luceneHnswScalarQuantizedVectorsFormat =
      BASE + codecs + "Lucene<version>HnswScalarQuantizedVectorsFormat";
  private static String luceneHnswScalarQuantizedVectorsFormatFallback =
      BASE + fallbackCodecs + "Lucene<version>HnswScalarQuantizedVectorsFormat";

  private static String luceneCodec = BASE + codecs + "Lucene<version>Codec";
  private static String luceneCodecFallback = BASE + fallbackCodecs + "Lucene<version>Codec";

  private static final Map<String, LuceneProvider> INSTANCES = new HashMap<>();

  private static MethodHandles.Lookup lookup = MethodHandles.lookup();

  private Class<?> flatVectorsFormat;
  private Class<?> hnswVectorsFormat;
  private Class<?> hnswVectorsReader;
  private Class<?> hnswVectorsWriter;
  private Class<?> binaryQuantizedVectorsFormat;
  private Class<?> hnswBinaryQuantizedVectorsFormat;
  private Class<?> scalarQuantizedVectorsFormat;
  private Class<?> hnswScalarQuantizedVectorsFormat;

  public static synchronized LuceneProvider getInstance(String version)
      throws ClassNotFoundException {
    LuceneProvider instance = INSTANCES.get(version);
    if (instance == null) {
      instance = new LuceneProvider(version);
      INSTANCES.put(version, instance);
    }
    return instance;
  }

  private LuceneProvider(String version) throws ClassNotFoundException {
    // TODO: Find a better way if possible, but as a separate initiative.
    if (LUCENE_102_BINARY_FORMAT_VERSION.equals(version)) {
      binaryQuantizedVectorsFormat =
          loadClass(
              setVersion(luceneBinaryQuantizedVectorsFormat, version),
              setVersion(luceneBinaryQuantizedVectorsFormatFallback, version));
      hnswBinaryQuantizedVectorsFormat =
          loadClass(
              setVersion(luceneHnswBinaryQuantizedVectorsFormat, version),
              setVersion(luceneHnswBinaryQuantizedVectorsFormatFallback, version));
      return;
    }

    flatVectorsFormat =
        loadClass(
            setVersion(luceneFlatVectorsFormat, version),
            setVersion(luceneFlatVectorsFormatFallback, version));
    hnswVectorsFormat =
        loadClass(
            setVersion(luceneHnswVectorsFormat, version),
            setVersion(luceneHnswVectorsFormatFallback, version));
    hnswVectorsReader =
        loadClass(
            setVersion(luceneHnswVectorsReader, version),
            setVersion(luceneHnswVectorsReaderFallback, version));
    hnswVectorsWriter =
        loadClass(
            setVersion(luceneHnswVectorsWriter, version),
            setVersion(luceneHnswVectorsWriterFallback, version));
    scalarQuantizedVectorsFormat =
        loadClass(
            setVersion(luceneScalarQuantizedVectorsFormat, version),
            setVersion(luceneScalarQuantizedVectorsFormatFallback, version));

    hnswScalarQuantizedVectorsFormat =
        loadClass(
            setVersion(luceneHnswScalarQuantizedVectorsFormat, version),
            setVersion(luceneHnswScalarQuantizedVectorsFormatFallback, version));
  }

  private static String setVersion(String pkg, String version) {
    return pkg.replaceAll("<version>", version);
  }

  private static Class<?> loadClass(String defaultClassName, String fallbackClassName)
      throws ClassNotFoundException {
    try {
      return Class.forName(defaultClassName);
    } catch (ClassNotFoundException defaultException) {
      // Load class from fallback package.
      try {
        return Class.forName(fallbackClassName);
      } catch (ClassNotFoundException fallbackException) {
        ClassNotFoundException missing =
            new ClassNotFoundException(
                "Unable to load Lucene class. Tried "
                    + defaultClassName
                    + " and "
                    + fallbackClassName);
        missing.addSuppressed(defaultException);
        missing.addSuppressed(fallbackException);
        throw missing;
      }
    }
  }

  public static Codec getCodec(String version)
      throws ClassNotFoundException,
          NoSuchMethodException,
          SecurityException,
          InstantiationException,
          IllegalAccessException,
          IllegalArgumentException,
          InvocationTargetException {
    Class<?> codecClass =
        loadClass(setVersion(luceneCodec, version), setVersion(luceneCodecFallback, version));
    Constructor<?> codecClassConstructor = codecClass.getConstructor();
    return (Codec) codecClassConstructor.newInstance();
  }

  public FlatVectorsFormat getLuceneFlatVectorsFormatInstance(FlatVectorsScorer scorer)
      throws Exception {
    try {
      Constructor<?> luceneFlatVectorsFormatConstructor =
          flatVectorsFormat.getConstructor(FlatVectorsScorer.class);
      return (FlatVectorsFormat) luceneFlatVectorsFormatConstructor.newInstance(scorer);
    } catch (Exception e) {
      log.log(Level.SEVERE, "Unable to initialize LuceneFlatVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  public KnnVectorsReader getLuceneHnswVectorsReaderInstance(
      SegmentReadState state, FlatVectorsReader reader) throws Exception {
    try {
      Constructor<?> luceneHnswVectorsReaderConstructor =
          hnswVectorsReader.getConstructor(SegmentReadState.class, FlatVectorsReader.class);
      return (KnnVectorsReader) luceneHnswVectorsReaderConstructor.newInstance(state, reader);
    } catch (Exception e) {
      log.log(Level.SEVERE, "Unable to initialize LuceneHnswVectorsReader: " + e.getMessage());
      throw e;
    }
  }

  public KnnVectorsWriter getLuceneHnswVectorsWriterInstance(
      SegmentWriteState state,
      int maxConn,
      int beamWidth,
      FlatVectorsWriter writer,
      int numMergeWorkers,
      TaskExecutor executor)
      throws Exception {
    try {
      Constructor<?> luceneHnswVectorsWriterConstructor =
          hnswVectorsWriter.getConstructor(
              SegmentWriteState.class,
              Integer.TYPE,
              Integer.TYPE,
              FlatVectorsWriter.class,
              Integer.TYPE,
              TaskExecutor.class);
      return (KnnVectorsWriter)
          luceneHnswVectorsWriterConstructor.newInstance(
              state, maxConn, beamWidth, writer, numMergeWorkers, executor);
    } catch (Exception e) {
      log.log(Level.SEVERE, "Unable to initialize LuceneHnswVectorsWriter: " + e.getMessage());
      throw e;
    }
  }

  public int getStaticIntParam(String param) throws ReflectiveOperationException {
    try {
      VarHandle varHandle = lookup.findStaticVarHandle(hnswVectorsFormat, param, Integer.TYPE);
      return (int) varHandle.get();
    } catch (NoSuchFieldException | IllegalAccessException e) {
      log.log(Level.SEVERE, "Unable to get " + param + ": " + e.getMessage());
      throw e;
    }
  }

  public List<VectorSimilarityFunction> getSimilarityFunctions()
      throws ReflectiveOperationException {
    try {
      VarHandle varHandle =
          lookup.findStaticVarHandle(hnswVectorsReader, "SIMILARITY_FUNCTIONS", List.class);
      return (List<VectorSimilarityFunction>) varHandle.get();
    } catch (NoSuchFieldException | IllegalAccessException e) {
      log.log(Level.SEVERE, "Unable to get SIMILARITY_FUNCTIONS: " + e.getMessage());
      throw e;
    }
  }

  /** Returns the Lucene 10.2 flat binary-quantized vectors format. */
  public FlatVectorsFormat getLuceneBinaryQuantizedVectorsFormatInstance() throws Exception {
    try {
      Constructor<?> luceneBinaryQuantizedVectorsFormatConstructor =
          binaryQuantizedVectorsFormat.getConstructor();
      return (FlatVectorsFormat) luceneBinaryQuantizedVectorsFormatConstructor.newInstance();
    } catch (Exception e) {
      log.log(
          Level.SEVERE,
          "Unable to initialize LuceneBinaryQuantizedVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  /**
   * Retains the original public spelling for source and binary compatibility.
   *
   * @deprecated Use {@link #getLuceneBinaryQuantizedVectorsFormatInstance()}.
   */
  @Deprecated(since = "26.10", forRemoval = false)
  public FlatVectorsFormat getluceneBinaryQuantizedVectorsFormatInstance() throws Exception {
    return getLuceneBinaryQuantizedVectorsFormatInstance();
  }

  /** Returns the Lucene 10.2 HNSW binary-quantized vectors format. */
  public KnnVectorsFormat getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(
      int maxConn, int beamWidth) throws Exception {
    try {
      Constructor<?> luceneHnswBinaryQuantizedVectorsFormatConstructor =
          hnswBinaryQuantizedVectorsFormat.getConstructor(int.class, int.class);
      return (KnnVectorsFormat)
          luceneHnswBinaryQuantizedVectorsFormatConstructor.newInstance(maxConn, beamWidth);
    } catch (Exception e) {
      log.log(
          Level.SEVERE,
          "Unable to initialize LuceneHnswBinaryQuantizedVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  /**
   * Retains the original, incorrectly typed JVM method descriptor.
   *
   * <p>The Lucene HNSW binary-quantized format is a {@link KnnVectorsFormat}, not a {@link
   * FlatVectorsFormat}; the former implementation therefore always failed its cast. Use {@link
   * #getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(int, int)}.
   *
   * @deprecated The original return type cannot represent Lucene's HNSW format.
   */
  @Deprecated(since = "26.10", forRemoval = false)
  public FlatVectorsFormat getLuceneHnswBinaryQuantizedVectorsFormatInstance(
      int maxConn, int beamWidth) throws Exception {
    throw new UnsupportedOperationException(
        "Lucene HNSW binary-quantized vectors require KnnVectorsFormat; use "
            + "getLuceneHnswBinaryQuantizedKnnVectorsFormatInstance(int, int)");
  }

  public FlatVectorsFormat getLuceneScalarQuantizedVectorsFormatInstance() throws Exception {
    try {
      Constructor<?> luceneScalarQuantizedVectorsFormatConstructor =
          scalarQuantizedVectorsFormat.getConstructor();
      return (FlatVectorsFormat) luceneScalarQuantizedVectorsFormatConstructor.newInstance();
    } catch (Exception e) {
      log.log(
          Level.SEVERE,
          "Unable to initialize LuceneScalarQuantizedVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  /**
   * Returns Lucene's HNSW scalar-quantized vectors format.
   *
   * @param maxConn maximum number of connections per graph node
   * @param beamWidth number of candidate neighbors tracked while building the graph
   * @return the configured scalar-quantized HNSW format
   * @throws Exception if the Lucene format cannot be constructed
   */
  public KnnVectorsFormat getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(
      int maxConn, int beamWidth) throws Exception {
    try {
      Constructor<?> luceneHnswScalarQuantizedVectorsFormatConstructor =
          hnswScalarQuantizedVectorsFormat.getConstructor(Integer.TYPE, Integer.TYPE);
      return (KnnVectorsFormat)
          luceneHnswScalarQuantizedVectorsFormatConstructor.newInstance(maxConn, beamWidth);
    } catch (Exception e) {
      log.log(
          Level.SEVERE,
          "Unable to initialize LuceneHnswScalarQuantizedVectorsFormat: " + e.getMessage());
      throw e;
    }
  }

  /**
   * Retains the original, incorrectly typed JVM method descriptor.
   *
   * <p>The Lucene HNSW scalar-quantized format is a {@link KnnVectorsFormat}, not a {@link
   * FlatVectorsFormat}; the former implementation therefore always failed its cast. Use {@link
   * #getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(int, int)}.
   *
   * @deprecated The original return type cannot represent Lucene's HNSW format.
   */
  @Deprecated(since = "26.10", forRemoval = false)
  public FlatVectorsFormat getLuceneHnswScalarQuantizedVectorsFormatInstance(
      int beamWidth, int maxConn) throws Exception {
    throw new UnsupportedOperationException(
        "Lucene HNSW scalar-quantized vectors require KnnVectorsFormat; use "
            + "getLuceneHnswScalarQuantizedKnnVectorsFormatInstance(int, int)");
  }
}
