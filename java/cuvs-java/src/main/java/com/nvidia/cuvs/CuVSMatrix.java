/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import com.nvidia.cuvs.spi.CuVSProvider;

/**
 * This represents a wrapper for a dataset to be used for index construction.
 * The purpose is to allow a caller to place the vectors into native memory
 * directly, instead of requiring the caller to load all the vectors into the heap
 * (e.g. with a float[][]).
 *
 * @since 25.06
 */
public interface CuVSMatrix extends AutoCloseable {

  enum DataType {
    FLOAT(4),
    HALF(2),
    INT(4),
    UINT(4),
    BYTE(1);

    private final int bytes;

    DataType(int bytes) {
      this.bytes = bytes;
    }

    public int bytes() {
      return bytes;
    }
  }

  enum MemoryKind {
    HOST,
    DEVICE
  }

  /**
   * Creates a dataset from an on-heap array of vectors.
   * This method will allocate an additional MemorySegment to hold the graph data.
   *
   * @since 25.08
   */
  static CuVSMatrix ofArray(float[][] vectors) {
    return CuVSProvider.provider().newMatrixFromArray(vectors);
  }

  /**
   * Creates a dataset from an on-heap array of vectors.
   * This method will allocate an additional MemorySegment to hold the graph data.
   *
   * @since 25.08
   */
  static CuVSMatrix ofArray(int[][] vectors) {
    return CuVSProvider.provider().newMatrixFromArray(vectors);
  }

  /**
   * Creates a dataset from an on-heap array of vectors.
   * This method will allocate an additional MemorySegment to hold the graph data.
   *
   * @since 25.08
   */
  static CuVSMatrix ofArray(byte[][] vectors) {
    return CuVSProvider.provider().newMatrixFromArray(vectors);
  }

  /**
   * A builder to construct a new matrix one row at a time
   * @param <T> the CuVSMatrix type to build
   */
  interface Builder<T extends CuVSMatrix> {
    /**
     * Adds a single vector to the matrix.
     *
     * @param vector A float array of as many elements as the dimensions
     */
    void addVector(float[] vector);

    /**
     * Adds a single vector to the matrix.
     *
     * @param vector A byte array of as many elements as the dimensions
     */
    void addVector(byte[] vector);

    /**
     * Adds a single vector to the matrix.
     *
     * @param vector An int array of as many elements as the dimensions
     */
    void addVector(int[] vector);

    /**
     * Adds a single vector to the matrix. Each element is a raw float16 bit pattern stored in a short.
     *
     * @param vector A short array of as many elements as the dimensions
     */
    void addVector(short[] vector);

    T build();
  }

  /**
   * Returns a builder to create a new instance of a host-memory matrix
   *
   * @param size      Number of rows (e.g. vectors in a dataset)
   * @param columns   Number of columns (e.g. dimension of each vector in the dataset)
   * @param dataType The data type of the dataset elements
   * @return a builder for creating a {@link CuVSHostMatrix}
   */
  static Builder<CuVSHostMatrix> hostBuilder(long size, long columns, DataType dataType) {
    return CuVSProvider.provider().newHostMatrixBuilder(size, columns, dataType);
  }

  /**
   * Returns a builder to create a new instance of a host-memory matrix
   *
   * @param size      Number of rows (e.g. vectors in a dataset)
   * @param columns   Number of columns (e.g. dimension of each vector in the dataset)
   * @param rowStride The stride (in number of elements) for each row. Must be -1 or > than {@code columns}
   * @param columnStride The stride for each column. Currently, it is not supported (must be -1)
   * @param dataType  The data type of the dataset elements
   * @return a builder for creating a {@link CuVSDeviceMatrix}
   */
  static Builder<CuVSHostMatrix> hostBuilder(
      long size, long columns, int rowStride, int columnStride, DataType dataType) {
    return CuVSProvider.provider()
        .newHostMatrixBuilder(size, columns, rowStride, columnStride, dataType);
  }

  /**
   * Returns a builder to create a new instance of a dataset
   *
   * @param resources CuVS resources used to allocate the device memory needed
   * @param size      Number of rows (e.g. vectors in a dataset)
   * @param columns   Number of columns (e.g. dimension of each vector in the dataset)
   * @param dataType  The data type of the dataset elements
   * @return a builder for creating a {@link CuVSDeviceMatrix}
   */
  static Builder<CuVSDeviceMatrix> deviceBuilder(
      CuVSResources resources, long size, long columns, DataType dataType) {
    return CuVSProvider.provider().newDeviceMatrixBuilder(resources, size, columns, dataType);
  }

  /**
   * Returns a builder to create a new instance of a dataset
   *
   * @param resources CuVS resources used to allocate the device memory needed
   * @param size      Number of rows (e.g. vectors in a dataset)
   * @param columns   Number of columns (e.g. dimension of each vector in the dataset)
   * @param rowStride The stride (in number of elements) for each row. Must be -1 or > than {@code columns}
   * @param columnStride The stride for each column. Currently, it is not supported (must be -1)
   * @param dataType  The data type of the dataset elements
   * @return a builder for creating a {@link CuVSDeviceMatrix}
   */
  static Builder<CuVSDeviceMatrix> deviceBuilder(
      CuVSResources resources,
      long size,
      long columns,
      int rowStride,
      int columnStride,
      DataType dataType) {
    return CuVSProvider.provider()
        .newDeviceMatrixBuilder(resources, size, columns, rowStride, columnStride, dataType);
  }

  /**
   * Gets the size of the dataset
   *
   * @return Size of the dataset
   */
  long size();

  /**
   * Gets the number of columns in the Dataset (e.g. the dimensions of the vectors in this dataset,
   * or the graph degree for the graph represented as a list of neighbours
   *
   * @return Dimensions of the vectors in the dataset
   */
  long columns();

  /**
   * Gets the element type
   *
   * @return a {@link DataType} describing the matrix element type
   */
  DataType dataType();

  /**
   * Get a view (0-copy) of the row data, as a list of integers (32 bit)
   *
   * @param row the row for which to return the data
   */
  RowView getRow(long row);

  /**
   * Copies the content of this dataset to an on-heap Java matrix (array of arrays).
   *
   * @param array the destination array. Must be of length {@link CuVSMatrix#size()} or bigger,
   *              and each element must be of length {@link CuVSMatrix#columns()} or bigger.
   */
  void toArray(int[][] array);

  /**
   * Copies the content of this dataset to an on-heap Java matrix (array of arrays).
   *
   * @param array the destination array. Must be of length {@link CuVSMatrix#size()} or bigger,
   *              and each element must be of length {@link CuVSMatrix#columns()} or bigger.
   */
  void toArray(float[][] array);

  /**
   * Copies the content of this dataset to an on-heap Java matrix (array of arrays).
   *
   * @param array the destination array. Must be of length {@link CuVSMatrix#size()} or bigger,
   *              and each element must be of length {@link CuVSMatrix#columns()} or bigger.
   */
  void toArray(byte[][] array);

  /**
   * Fills the provided, pre-allocated host matrix with data from this matrix.
   * The content of the provided host matrix will be overwritten; the 2 matrices must have the
   * same element type and dimension.
   *
   * @param hostMatrix  the host-memory-backed matrix to fill.
   */
  void toHost(CuVSHostMatrix hostMatrix);

  /**
   * Returns a host matrix; if the matrix is already a host matrix, a "weak" reference to the same host memory
   * is returned. If the matrix is a device matrix, a newly allocated matrix will be populated with data from
   * the device matrix.
   * The returned host matrix will need to be managed by the caller, which will be
   * responsible to call {@link CuVSMatrix#close()} to free its resources when done.
   */
  CuVSHostMatrix toHost();

  /**
   * Fills the provided, pre-allocated device matrix with data from this matrix.
   * The content of the provided device matrix will be overwritten; the 2 matrices must have the
   * same element type and dimension.
   *
   * @param deviceMatrix  the device-memory-backed matrix to fill.
   */
  void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources);

  /**
   * Returns a device matrix; if this matrix is already a device matrix, a "weak" reference to the same host memory
   * is returned. If the matrix is a host matrix, a newly allocated matrix will be populated with data from
   * the host matrix.
   * The returned device matrix will need to be managed by the caller, which will be
   * responsible to call {@link CuVSMatrix#close()} to free its resources when done.
   */
  CuVSDeviceMatrix toDevice(CuVSResources cuVSResources);

  @Override
  void close();
}
