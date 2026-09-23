/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.util.InfoStream;

/**
 * This class provides common static utility methods.
 *
 * @since 25.10
 */
public class Utils {

  static final Logger log = Logger.getLogger(Utils.class.getName());

  /**
   * A utility method that rethrows known throwable types without changing their identity.
   *
   * <p>In particular, {@link Error} instances must not be converted to a {@link
   * RuntimeException}; callers rely on errors retaining their original type and stack trace.
   *
   * <p>This method never returns normally; its return type exists solely so callers can write
   * {@code throw handleThrowable(t);}, letting the compiler verify that the enclosing statement
   * always completes abruptly.
   *
   * @param t the throwable object
   * @return never returns; always throws
   * @throws IOException
   */
  static RuntimeException handleThrowable(Throwable t) throws IOException {
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
   * Builds a host-memory CuVSMatrix from a list of float vectors.
   *
   * <p>Copies vectors directly into native host memory without creating an intermediate {@code
   * float[][]} on the heap.
   *
   * @param data The float vectors
   * @param dimensions The number of float elements in each vector
   * @return a host-memory CuVSMatrix
   */
  static CuVSHostMatrix createHostFloatMatrix(List<float[]> data, int dimensions) {
    try (CuVSMatrix.Builder<CuVSHostMatrix> builder =
        CuVSMatrix.hostBuilder(data.size(), dimensions, CuVSMatrix.DataType.FLOAT)) {
      for (float[] vector : data) {
        builder.addVector(vector);
      }
      return builder.build();
    }
  }

  /**
   * Builds a host-memory CuVSMatrix from byte vectors without first materializing the list as an
   * intermediate {@code byte[][]}.
   */
  static CuVSHostMatrix createHostByteMatrix(List<byte[]> data, int bytesPerVector) {
    try (CuVSMatrix.Builder<CuVSHostMatrix> builder =
        CuVSMatrix.hostBuilder(data.size(), bytesPerVector, CuVSMatrix.DataType.BYTE)) {
      for (byte[] vector : data) {
        builder.addVector(vector);
      }
      return builder.build();
    }
  }

  /** Builds a host-memory CuVSMatrix from a 2D byte array. */
  static CuVSHostMatrix createHostByteMatrixFromArray(byte[][] data, int bytesPerVector) {
    try (CuVSMatrix.Builder<CuVSHostMatrix> builder =
        CuVSMatrix.hostBuilder(data.length, bytesPerVector, CuVSMatrix.DataType.BYTE)) {
      for (byte[] vector : data) {
        builder.addVector(vector);
      }
      return builder.build();
    }
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
