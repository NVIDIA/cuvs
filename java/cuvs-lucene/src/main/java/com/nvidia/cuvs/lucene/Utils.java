/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.util.InfoStream;

/**
 * This class provides common static utility methods.
 *
 * @since 25.10
 */
public class Utils {

  static final Logger log = Logger.getLogger(Utils.class.getName());

  /**
   * A utility method that throws specific types of throwable objects based on types.
   *
   * @param t the throwable object
   * @throws IOException
   */
  static void handleThrowable(Throwable t) throws IOException {
    switch (t) {
      case IOException ioe -> throw ioe;
      case Error error -> throw error;
      case RuntimeException re -> throw re;
      case null, default -> throw new RuntimeException("UNEXPECTED: exception type", t);
    }
  }

  /**
   * Builds a host-memory CuVSMatrix from a list of float vectors.
   *
   * <p>Copies vectors directly into a native host matrix via {@link CuVSMatrix#hostBuilder},
   * without creating an intermediate {@code float[][]} on the heap.
   *
   * @param data The float vectors
   * @param dimensions The number of float elements in each vector
   * @return a host-memory CuVSMatrix
   */
  static CuVSMatrix createFloatMatrix(List<float[]> data, int dimensions) {
    CuVSMatrix.Builder<?> builder =
        CuVSMatrix.hostBuilder(data.size(), dimensions, CuVSMatrix.DataType.FLOAT);
    for (float[] vector : data) {
      builder.addVector(vector);
    }
    return builder.build();
  }

  /**
   * Builds a host-memory CuVSMatrix from a list of byte vectors (e.g. quantized vectors).
   *
   * @param data The byte vectors (packed bits for binary quantization)
   * @param bytesPerVector The number of bytes in each vector
   * @return a host-memory CuVSMatrix with BYTE data type
   */
  static CuVSMatrix createByteMatrix(List<byte[]> data, int bytesPerVector) {
    CuVSMatrix.Builder<?> builder =
        CuVSMatrix.hostBuilder(data.size(), bytesPerVector, CuVSMatrix.DataType.BYTE);
    for (byte[] vector : data) {
      builder.addVector(vector);
    }
    return builder.build();
  }

  /**
   * Builds a host-memory CuVSMatrix from a 2D byte array (e.g. quantized vectors).
   *
   * @param data The 2D byte array (packed bits for binary quantization)
   * @param bytesPerVector The number of bytes in each vector
   * @return a host-memory CuVSMatrix with BYTE data type
   */
  static CuVSMatrix createByteMatrixFromArray(byte[][] data, int bytesPerVector) {
    CuVSMatrix.Builder<?> builder =
        CuVSMatrix.hostBuilder(data.length, bytesPerVector, CuVSMatrix.DataType.BYTE);
    for (byte[] vector : data) {
      builder.addVector(vector);
    }
    return builder.build();
  }

  /**
   * A utility method to convert nanoseconds to milliseconds.
   *
   * @param nanos
   * @return milliseconds
   */
  static long nanosToMillis(long nanos) {
    return Duration.ofNanos(nanos).toMillis();
  }

  /**
   * Creates an instance of CuVSResources.
   *
   * @return an instance of CuVSResources
   */
  static CuVSResources cuVSResourcesOrNull() {
    try {
      System.loadLibrary("cudart");
    } catch (UnsatisfiedLinkError e) {
      log.log(Level.WARNING, "Could not load CUDA runtime library: " + e.getMessage());
    }
    try {
      return CuVSResources.create();
    } catch (UnsupportedOperationException uoe) {
      log.log(
          Level.WARNING,
          "cuVS is not supported on this platform or java version: " + uoe.getMessage());
    } catch (Throwable t) {
      if (t instanceof ExceptionInInitializerError ex) {
        t = ex.getCause();
      }
      log.log(Level.WARNING, "Exception occurred during creation of cuVS resources. " + t);
    }
    return null;
  }

  /**
   * A utility method that conditionally ignores certain throwable objects
   *
   * @param t the throwable object
   * @param msg the message to check
   * @throws IOException
   */
  static void handleThrowableWithIgnore(Throwable t, String msg) throws IOException {
    if (t.getMessage().contains(msg)) {
      return;
    }
    handleThrowable(t);
  }

  /**
   * Creates a list of float vectors from the input
   *
   * @param mergedVectorValues instance of {@link FloatVectorValues}
   * @return a list of float arrays
   * @throws IOException I/O Exception
   */
  static List<float[]> createListFromMergedVectors(FloatVectorValues mergedVectorValues)
      throws IOException {
    List<float[]> vectors = new ArrayList<float[]>();
    KnnVectorValues.DocIndexIterator iter = mergedVectorValues.iterator();
    for (int docV = iter.nextDoc(); docV != NO_MORE_DOCS; docV = iter.nextDoc()) {
      float[] vector = mergedVectorValues.vectorValue(iter.index());
      vectors.add(vector.clone());
    }
    return vectors;
  }

  /**
   * Utility to print info/debug messages via InfoStream.
   *
   * @param infoStream the writer's infostream
   * @param component the name of the index writer
   * @param msg the log message to push via the InfoStream
   */
  static void info(InfoStream infoStream, String component, String msg) {
    if (infoStream.isEnabled(component)) {
      infoStream.message(component, msg);
    }
  }
}
