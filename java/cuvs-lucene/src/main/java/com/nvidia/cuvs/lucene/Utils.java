/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CuVSDeviceMatrix;
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

  /*
   * Builds a host-memory CuVSMatrix from a list of float vectors.
   *
   * Copies vectors directly into a native host matrix without creating an intermediate float[][]
   * on the heap.
   */
  static CuVSHostMatrix createHostFloatMatrix(List<float[]> data, int dimensions)
      throws IOException {
    return MatrixBuilderLifecycle.build(
        CuVSMatrix.hostBuilder(data.size(), dimensions, CuVSMatrix.DataType.FLOAT),
        builder -> {
          for (float[] vector : data) {
            builder.addVector(vector);
          }
          return builder.build();
        });
  }

  /* Builds a device-memory matrix from a list of float vectors. */
  static CuVSDeviceMatrix createDeviceFloatMatrix(
      List<float[]> data, int dimensions, CuVSResources resources) throws IOException {
    return MatrixBuilderLifecycle.build(
        CuVSMatrix.deviceBuilder(resources, data.size(), dimensions, CuVSMatrix.DataType.FLOAT),
        builder -> {
          for (float[] vector : data) {
            builder.addVector(vector);
          }
          return builder.build();
        });
  }

  /*
   * Closes an index that owns {@code dataset}. If index cleanup fails before releasing the
   * dataset, a direct dataset close is attempted and attached to the index failure when needed.
   */
  static void closeIndexWithDatasetFallback(AutoCloseable index, AutoCloseable dataset)
      throws Exception {
    try {
      index.close();
    } catch (Throwable indexCloseFailure) {
      try {
        dataset.close();
      } catch (Throwable datasetCloseFailure) {
        if (indexCloseFailure != datasetCloseFailure) {
          indexCloseFailure.addSuppressed(datasetCloseFailure);
        }
      }
      rethrowCloseFailure(indexCloseFailure);
    }
  }

  /* Starts an ownership scope for a dataset that may later be transferred to an index. */
  static <I extends AutoCloseable> OwnedIndex<I> ownDataset(AutoCloseable dataset) {
    return new OwnedIndex<>(dataset);
  }

  /*
   * Owns a dataset immediately and, after {@link #transferTo}, closes the owning index with a
   * direct dataset-close fallback.
   */
  static final class OwnedIndex<I extends AutoCloseable> implements AutoCloseable {
    private AutoCloseable dataset;
    private I index;
    private boolean closed;

    private OwnedIndex(AutoCloseable dataset) {
      this.dataset = java.util.Objects.requireNonNull(dataset, "dataset");
    }

    void transferTo(I index) {
      if (closed || this.index != null) {
        throw new IllegalStateException("Dataset ownership has already been transferred");
      }
      this.index = java.util.Objects.requireNonNull(index, "index");
    }

    @Override
    public void close() throws Exception {
      if (closed) {
        return;
      }
      closed = true;
      AutoCloseable ownedDataset = dataset;
      I ownedIndex = index;
      dataset = null;
      index = null;
      if (ownedIndex == null) {
        ownedDataset.close();
      } else {
        closeIndexWithDatasetFallback(ownedIndex, ownedDataset);
      }
    }
  }

  private static void rethrowCloseFailure(Throwable failure) throws Exception {
    if (failure instanceof Exception exception) {
      throw exception;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    throw new AssertionError("Unexpected throwable from AutoCloseable.close()", failure);
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
