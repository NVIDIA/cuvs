/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
   * A method to build a CuVSMatrix from a list of float vectors.
   *
   * Uses CuVSMatrix.Builder to copy vectors directly to device memory
   * without creating intermediate heap arrays.
   *
   * @param data The float vectors
   * @param dimensions The number float elements in each vector
   * @param resources The CuVS resources for device matrix creation
   * @return an instance of CuVSMatrix
   */
  static CuVSMatrix createFloatMatrix(List<float[]> data, int dimensions, CuVSResources resources) {
    // Use Builder pattern to avoid intermediate float[][] allocation
    // and copy directly from List to device memory
    CuVSMatrix.Builder<?> builder =
        CuVSMatrix.deviceBuilder(
            resources,
            data.size(), // rows (number of vectors)
            dimensions, // columns (vector dimension)
            CuVSMatrix.DataType.FLOAT);

    // Add vectors one by one - builder copies directly to device memory
    for (float[] vector : data) {
      builder.addVector(vector);
    }

    return builder.build();
  }

  /**
   * A method to build a CuVSMatrix from a list of byte vectors (for binary quantized vectors).
   *
   * Uses CuVSMatrix.Builder to copy vectors directly to device memory
   * without creating intermediate heap arrays.
   *
   * @param data The byte vectors (packed bits for binary quantization)
   * @param bytesPerVector The number of bytes in each vector
   * @param resources The CuVS resources for device matrix creation
   * @return an instance of CuVSMatrix with BYTE data type
   */
  static CuVSMatrix createByteMatrix(
      List<byte[]> data, int bytesPerVector, CuVSResources resources) {
    // Use Builder pattern to avoid intermediate byte[][] allocation
    // and copy directly from List to device memory
    CuVSMatrix.Builder<?> builder =
        CuVSMatrix.deviceBuilder(
            resources,
            data.size(), // rows (number of vectors)
            bytesPerVector, // columns (bytes per vector)
            CuVSMatrix.DataType.BYTE);

    // Add vectors one by one - builder copies directly to device memory
    for (byte[] vector : data) {
      builder.addVector(vector);
    }

    return builder.build();
  }

  /**
   * A method to build a CuVSMatrix from a 2D byte array (for binary quantized vectors).
   *
   * @param data The 2D byte array (packed bits for binary quantization)
   * @param bytesPerVector The number of bytes in each vector
   * @param resources The CuVS resources for device matrix creation
   * @return an instance of CuVSMatrix with BYTE data type
   */
  static CuVSMatrix createByteMatrixFromArray(
      byte[][] data, int bytesPerVector, CuVSResources resources) {
    CuVSMatrix.Builder<?> builder =
        CuVSMatrix.deviceBuilder(
            resources,
            data.length, // rows (number of vectors)
            bytesPerVector, // columns (bytes per vector)
            CuVSMatrix.DataType.BYTE);

    // Add vectors one by one - builder copies directly to device memory
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
