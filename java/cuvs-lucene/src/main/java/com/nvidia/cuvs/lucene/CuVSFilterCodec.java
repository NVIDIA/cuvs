/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.LibraryException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FieldInfosFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.LiveDocsFormat;
import org.apache.lucene.codecs.NormsFormat;
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PostingsFormat;
import org.apache.lucene.codecs.SegmentInfoFormat;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.TermVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.util.Version;

/**
 * Base class of the cuvs-lucene codecs: a {@link FilterCodec} that replaces its delegate's
 * {@link KnnVectorsFormat}.
 *
 * <p>Each cuvs-lucene codec wraps the default codec of one Lucene release, for example
 * {@code Lucene101Codec} for Lucene 10.2. Use {@link CuVSCodecs} to get the codecs for the Lucene
 * release in use.
 */
abstract class CuVSFilterCodec extends FilterCodec {

  private static final Logger log = Logger.getLogger(CuVSFilterCodec.class.getName());
  // Every registered codec runs into the same problem, so it is logged only once.
  private static final AtomicBoolean unavailableLogged = new AtomicBoolean();

  private KnnVectorsFormat format;

  /**
   * Creates a codec.
   *
   * @param name the codec's name
   * @param delegate the codec to use for everything but vectors; for the codecs of this module,
   *     the result of {@link #delegate(String, Supplier)}
   * @param format creates the {@link KnnVectorsFormat} to use for vectors
   */
  protected CuVSFilterCodec(String name, Codec delegate, Supplier<KnnVectorsFormat> format) {
    super(name, delegate);
    try {
      setKnnVectorsFormat(format.get());
    } catch (LibraryException | LinkageError e) {
      // Lucene's service loader creates every codec eagerly, and an exception here would break
      // every codec lookup in the JVM. A format that does not link on the Lucene release in use, or
      // cannot reach the cuVS library, costs only this codec, which then fails when used, like an
      // unavailable delegate. Other exceptions, such as invalid arguments, reach the caller.
      String problem = "Its vectors format is not available: " + e;
      logUnavailable(name, problem);
      setKnnVectorsFormat(new UnavailableFormat(name, problem));
    }
  }

  /**
   * Returns the Lucene codec a cuvs-lucene codec delegates to, or, if it cannot be created on the
   * Lucene release in use, a codec that fails with an explanation when it is used.
   *
   * <p>Lucene's service loader creates every registered codec when it first looks one up, and an
   * exception there breaks every codec lookup in the JVM, Lucene's own included. A cuvs-lucene
   * artifact running on another Lucene release than it was built for must therefore still create
   * its codecs, and only fail once they are used.
   *
   * <p>Pass {@code delegate} as a lambda, such as {@code () -> LuceneCompat.lucene101Codec()}, not
   * as a method reference: the JVM resolves a method reference, and so loads {@code LuceneCompat},
   * before this method runs, which is exactly what can fail on the wrong Lucene release.
   *
   * <p>The first time a codec is unavailable, the problem is logged at {@code SEVERE}, so that it
   * shows up when Lucene first looks up its codecs, typically at startup, even though it only fails
   * once the codec is used. Applications that would rather not start can call {@link
   * CuVSCodecs#checkLuceneVersion()}.
   *
   * @param name the name of the cuvs-lucene codec, for the error message
   * @param delegate creates the Lucene codec to delegate to
   * @return the Lucene codec, or a codec that throws {@link IllegalStateException} when used
   */
  static Codec delegate(String name, Supplier<Codec> delegate) {
    String problem = LuceneVersionGuard.mismatchMessage(Version.LATEST);
    if (problem == null) {
      try {
        return delegate.get();
      } catch (LinkageError e) {
        problem = "The Lucene codec " + name + " delegates to is not available: " + e;
      }
    }
    logUnavailable(name, problem);
    return new UnavailableCodec(name, problem);
  }

  /** Logs, once per JVM, that a codec cannot be used. */
  private static void logUnavailable(String name, String problem) {
    if (unavailableLogged.compareAndSet(false, true)) {
      log.log(
          Level.SEVERE,
          "The cuvs-lucene codecs cannot be used, starting with "
              + name
              + ": "
              + problem
              + " Lucene and indexes that do not use these codecs are not affected; reading or"
              + " writing an index that does fails.");
    }
  }

  /**
   * Sets whether the problem {@link #delegate} logs has already been logged, and returns the
   * previous value. The flag is static, so a test that goes through {@link #delegate}'s failure path
   * restores it afterwards, leaving later tests in the same JVM unaffected.
   */
  static boolean setUnavailableLoggedForTesting(boolean logged) {
    return unavailableLogged.getAndSet(logged);
  }

  /**
   * Get the configured {@link KnnVectorsFormat}.
   *
   * @return the instance of the {@link KnnVectorsFormat}
   */
  @Override
  public KnnVectorsFormat knnVectorsFormat() {
    return format;
  }

  /**
   * Sets the {@link KnnVectorsFormat} this codec uses for vectors.
   *
   * @param format the {@link KnnVectorsFormat} to set
   */
  public void setKnnVectorsFormat(KnnVectorsFormat format) {
    this.format = format;
  }

  /**
   * Set the {@link KnnVectorsFormat}.
   *
   * @param format the {@link KnnVectorsFormat} to set
   * @deprecated use {@code setKnnVectorsFormat(KnnVectorsFormat)}, which matches the name of
   *     {@link #knnVectorsFormat()}
   */
  @Deprecated(since = "26.12", forRemoval = true)
  public void setKnnFormat(KnnVectorsFormat format) {
    setKnnVectorsFormat(format);
  }

  /** Stands in for a delegate codec that cannot be created, and throws when any format is used. */
  private static final class UnavailableCodec extends Codec {

    private final String problem;

    UnavailableCodec(String name, String problem) {
      super(name);
      this.problem = problem;
    }

    private IllegalStateException unavailable() {
      return new IllegalStateException(getName() + " cannot be used: " + problem);
    }

    @Override
    public PostingsFormat postingsFormat() {
      throw unavailable();
    }

    @Override
    public DocValuesFormat docValuesFormat() {
      throw unavailable();
    }

    @Override
    public StoredFieldsFormat storedFieldsFormat() {
      throw unavailable();
    }

    @Override
    public TermVectorsFormat termVectorsFormat() {
      throw unavailable();
    }

    @Override
    public FieldInfosFormat fieldInfosFormat() {
      throw unavailable();
    }

    @Override
    public SegmentInfoFormat segmentInfoFormat() {
      throw unavailable();
    }

    @Override
    public NormsFormat normsFormat() {
      throw unavailable();
    }

    @Override
    public LiveDocsFormat liveDocsFormat() {
      throw unavailable();
    }

    @Override
    public CompoundFormat compoundFormat() {
      throw unavailable();
    }

    @Override
    public PointsFormat pointsFormat() {
      throw unavailable();
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
      throw unavailable();
    }
  }

  /** Stands in for a vectors format that cannot be created, and throws when it is used. */
  private static final class UnavailableFormat extends KnnVectorsFormat {

    private final String problem;

    UnavailableFormat(String codecName, String problem) {
      super(codecName);
      this.problem = problem;
    }

    private IllegalStateException unavailable() {
      return new IllegalStateException(getName() + " cannot be used: " + problem);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) {
      throw unavailable();
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) {
      throw unavailable();
    }

    @Override
    public int getMaxDimensions(String fieldName) {
      throw unavailable();
    }
  }
}
