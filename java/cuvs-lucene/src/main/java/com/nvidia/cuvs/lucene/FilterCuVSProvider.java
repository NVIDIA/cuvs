/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CagraIndexParams;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraIndexParams.HnswHeuristicType;
import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSMatrix.Builder;
import com.nvidia.cuvs.CuVSMatrix.DataType;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.GPUInfoProvider;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.HnswIndexParams;
import com.nvidia.cuvs.TieredIndex;
import com.nvidia.cuvs.spi.CuVSProvider;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.logging.Level;

class FilterCuVSProvider implements CuVSProvider {

  private final CuVSProvider delegate;

  FilterCuVSProvider(CuVSProvider delegate) {
    this.delegate = delegate;
  }

  @Override
  public Path nativeLibraryPath() {
    return CuVSProvider.TMPDIR;
  }

  @Override
  public CuVSResources newCuVSResources(Path tempPath) throws Throwable {
    return delegate.newCuVSResources(tempPath);
  }

  @Override
  public BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException {
    return delegate.newBruteForceIndexBuilder(cuVSResources);
  }

  @Override
  public CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException {
    return delegate.newCagraIndexBuilder(cuVSResources);
  }

  @Override
  public HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException {
    return delegate.newHnswIndexBuilder(cuVSResources);
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] arg0) throws Throwable {
    return delegate.mergeCagraIndexes(arg0);
  }

  @Override
  public GPUInfoProvider gpuInfoProvider() {
    return delegate.gpuInfoProvider();
  }

  @Override
  public Builder<CuVSHostMatrix> newHostMatrixBuilder(long rows, long cols, DataType dataType) {
    return delegate.newHostMatrixBuilder(rows, cols, dataType);
  }

  @Override
  public Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long rows, long cols, int maxRows, int maxCols, DataType dataType) {
    return delegate.newHostMatrixBuilder(rows, cols, maxRows, maxCols, dataType);
  }

  @Override
  public Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources resources, long rows, long cols, DataType dataType) {
    return delegate.newDeviceMatrixBuilder(resources, rows, cols, dataType);
  }

  @Override
  public Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources resources, long rows, long cols, int maxRows, int maxCols, DataType dataType) {
    return delegate.newDeviceMatrixBuilder(resources, rows, cols, maxRows, maxCols, dataType);
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    return delegate.newNativeMatrixBuilder();
  }

  @Override
  public MethodHandle newNativeMatrixBuilderWithStrides() {
    return delegate.newNativeMatrixBuilderWithStrides();
  }

  @Override
  public CuVSMatrix newMatrixFromArray(float[][] vectors) {
    return delegate.newMatrixFromArray(vectors);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(int[][] vectors) {
    return delegate.newMatrixFromArray(vectors);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(byte[][] vectors) {
    return delegate.newMatrixFromArray(vectors);
  }

  @Override
  public TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException {
    return delegate.newTieredIndexBuilder(cuVSResources);
  }

  @Override
  public CagraIndexParams cagraIndexParamsFromHnswParams(
      long arg0, long arg1, int arg2, int arg3, HnswHeuristicType arg4, CuvsDistanceType arg5) {
    return delegate.cagraIndexParamsFromHnswParams(arg0, arg1, arg2, arg3, arg4, arg5);
  }

  @Override
  public Level getLogLevel() {
    return delegate.getLogLevel();
  }

  @Override
  public void setLogLevel(Level arg0) {
    delegate.setLogLevel(arg0);
  }

  @Override
  public HnswIndex hnswIndexFromCagra(HnswIndexParams arg0, CagraIndex arg1) throws Throwable {
    return delegate.hnswIndexFromCagra(arg0, arg1);
  }

  @Override
  public void enableRMMManagedPooledMemory(int arg0, int arg1) {
    delegate.enableRMMManagedPooledMemory(arg0, arg1);
  }

  @Override
  public void enableRMMPooledMemory(int arg0, int arg1) {
    delegate.enableRMMPooledMemory(arg0, arg1);
  }

  @Override
  public void resetRMMPooledMemory() {
    delegate.resetRMMPooledMemory();
  }
}
