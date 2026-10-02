/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.LibraryException;
import java.io.IOException;
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
 * {@code Lucene101Codec} for Lucene 10.2; that codec's version number is the cuvs-lucene codec's
 * <em>generation</em>. Once a newer Lucene release replaces that default codec, Lucene moves the
 * old one to its backward codecs, which can read but not write. A cuvs-lucene codec of that
 * generation is kept so that existing indexes stay readable, and like the codec it wraps it refuses
 * to write: only the generation {@code LuceneCompat.CURRENT_CODEC_GENERATION} names writes. Use
 * {@link CuVSCodecs} to get the codecs that can write on the Lucene release in use.
 */
abstract class CuVSFilterCodec extends FilterCodec {

  private static final Logger log = Logger.getLogger(CuVSFilterCodec.class.getName());
  // Every registered codec runs into the same problem, so it is logged only once.
  private static final AtomicBoolean unavailableLogged = new AtomicBoolean();

  private final boolean readOnly;
  private KnnVectorsFormat format;

  /**
   * Creates one of cuvs-lucene's own codecs, which wraps the default codec of a Lucene release and
   * can only write on the Lucene release where that codec is still the default.
   *
   * @param name the codec's name
   * @param generation the version number of the Lucene codec it wraps, such as 101 for
   *     {@code Lucene101Codec}
   * @param delegate creates that Lucene codec, as a lambda; see {@link #delegate(String, Supplier)}
   * @param format creates the {@link KnnVectorsFormat} to use for vectors
   */
  protected CuVSFilterCodec(
      String name, int generation, Supplier<Codec> delegate, Supplier<KnnVectorsFormat> format) {
    this(
        name,
        delegate(name, delegate),
        generation != LuceneCompat.CURRENT_CODEC_GENERATION,
        format);
  }

  /**
   * Creates a codec with a delegate chosen by the caller, which is never made read-only: whether
   * the delegate can write is the caller's responsibility.
   *
   * @param name the codec's name
   * @param delegate the codec to use for everything but vectors
   * @param format creates the {@link KnnVectorsFormat} to use for vectors
   */
  protected CuVSFilterCodec(String name, Codec delegate, Supplier<KnnVectorsFormat> format) {
    this(name, delegate, false, format);
  }

  private CuVSFilterCodec(
      String name, Codec delegate, boolean readOnly, Supplier<KnnVectorsFormat> format) {
    super(name, delegate);
    this.readOnly = readOnly;
    try {
      setKnnFormat(format.get());
    } catch (LibraryException ex) {
      log.log(
          Level.SEVERE,
          "Couldn't load native library, possible classloader issue. " + ex.getMessage());
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
   * @param name the name of the cuvs-lucene codec, for the error message
   * @param delegate creates the Lucene codec to delegate to
   * <p>The first time a codec is unavailable, the problem is logged at {@code SEVERE}, so that it
   * shows up when Lucene first looks up its codecs, typically at startup, even though it only fails
   * once the codec is used. Applications that would rather not start can call {@link
   * CuVSCodecs#checkLuceneVersion()}.
   *
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
    return new UnavailableCodec(name, problem);
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
   * Set the {@link KnnVectorsFormat}.
   *
   * @param format the {@link KnnVectorsFormat} to set
   */
  public void setKnnFormat(KnnVectorsFormat format) {
    // A format already made read-only, such as one read back from knnVectorsFormat(), is kept as
    // is.
    boolean wrap = readOnly && format != null && format instanceof ReadOnlyFormat == false;
    this.format = wrap ? new ReadOnlyFormat(getName(), format) : format;
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

  /** Reads through the wrapped format, and refuses to write. */
  private static final class ReadOnlyFormat extends KnnVectorsFormat {

    private final String codecName;
    private final KnnVectorsFormat in;

    ReadOnlyFormat(String codecName, KnnVectorsFormat in) {
      super(in.getName());
      this.codecName = codecName;
      this.in = in;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
      throw new UnsupportedOperationException(
          "The "
              + codecName
              + " codec can only read indexes on Lucene "
              + Version.LATEST
              + ". Use the codecs from CuVSCodecs to write.");
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
      return in.fieldsReader(state);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
      return in.getMaxDimensions(fieldName);
    }

    @Override
    public String toString() {
      return "ReadOnly(" + in + ")";
    }
  }
}
